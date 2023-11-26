from jax import jit, vmap, jacrev, custom_vjp, vjp, tree_map, value_and_grad
from jax import lax
from functools import partial
from jax.tree_util import tree_flatten
import jax.numpy as jnp
from inference.MP_Inference import lds_expected_stats_from_potentials, lds_to_hmm_mf, hmm_to_lds_mf, hmm_inference, lds_inference, hmm_kl, lds_kl, lds_kl_surr, lds_transition_params_to_nat, hmm_kl_full, lds_kl_full, lds_kl_gen, hmm_kl_gen, lds_inference_and_sample
from utils import flatten_es, unflatten_es, make_csr, gaus_param_min
from jax.experimental.host_callback import call, id_print, id_tap
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve
import time
import numpy as np
from jax.random import dirichlet
import jax
from jax.experimental import host_callback
from utils import wandb_log, wandb_log_internal
from inference.MP_Inference import lds_expected_stats_from_potentials, jumping_lds_to_hmm_mf, jumping_hmm_to_lds_mf, trans_hmm_inference, lds_inference, trans_hmm_kl, lds_kl, lds_transition_params_to_nat, trans_hmm_kl_full, lds_kl_full

MAX_ITER = 10
RICHARDSON_CLIPPING_THRESH = 1e2

def jslds_inference_implicit(recog_potentials, lds_params, hmm_params, initializer):
    if initializer.shape == (2,):
        N, K = recog_potentials[0].shape[0], lds_params[0][0].shape[0]
        marginals = dirichlet(initializer, jnp.ones(K)*0.1, shape=(N,))
#         cat_expected_stats = jnp.ones((N,K))/K
    else:
        marginals = initializer

    init_es = jnp.expand_dims(marginals[0],-1).astype(recog_potentials[0].dtype)
    trans_es = vmap(jnp.outer)(marginals[:-1], marginals[1:]).astype(recog_potentials[0].dtype)
    cat_expected_stats = (init_es, trans_es)

    def cond_fun(vals):
        i, kl, old_kl, _, _ = vals
        return jnp.logical_and(abs(kl - old_kl)/old_kl >= 1e-5, i < MAX_ITER)

    def body_fun(vals):
        i, kl, _, _, cat_expected_stats = vals
        gaus_natparam, E_prior_logZ = jumping_hmm_to_lds_mf(cat_expected_stats, *lds_params)
        gaus_expected_stats, gaus_logZ, _ = lds_inference(recog_potentials, *gaus_natparam)
        gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, gaus_logZ)

        cat_natparam = jumping_lds_to_hmm_mf(gaus_expected_stats, *lds_params)
        cat_expected_stats, hmm_logZ = trans_hmm_inference(*tree_map(lambda x,y: x+y, cat_natparam, hmm_params))
        cat_kl = trans_hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)

        return i+1, gaus_kl + cat_kl, kl, gaus_expected_stats, cat_expected_stats

    all_vals = body_fun((0, 1e10, jnp.inf, None, cat_expected_stats))
    i, _, _, gaus_expected_stats, cat_expected_stats = lax.while_loop(cond_fun, body_fun, all_vals)
#     id_print(i)
    return gaus_expected_stats, cat_expected_stats

@jit
def jslds_kl(recog_potentials, lds_params, hmm_params, 
             gaus_expected_stats, cat_expected_stats, lds_logZ):
    gaus_natparam, E_prior_logZ = jumping_hmm_to_lds_mf(cat_expected_stats, *lds_params)
    cat_natparam = jumping_lds_to_hmm_mf(gaus_expected_stats, *lds_params)

    _, hmm_logZ = trans_hmm_inference(*tree_map(lambda x,y: x+y, cat_natparam, hmm_params))

    cat_kl = trans_hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
    gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, lds_logZ)
    return gaus_kl + cat_kl

@jit
def jslds_kl_sur(recog_potentials, lds_params, hmm_params, 
                 gaus_expected_stats, cat_expected_stats, lds_logZ):
    gaus_natparam, E_prior_logZ = jumping_hmm_to_lds_mf(cat_expected_stats, *lds_params)
    cat_natparam = jumping_lds_to_hmm_mf(gaus_expected_stats, *lds_params)

    _, hmm_logZ = trans_hmm_inference(*tree_map(lambda x,y: x+y, cat_natparam, hmm_params))

    gaus_kl = lds_kl_surr(recog_potentials, gaus_expected_stats, E_prior_logZ, lds_logZ)
    cat_kl = trans_hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
    return gaus_kl + cat_kl

@jit
def jslds_kl_det(recog_potentials, prior_params, inference_params, 
                 gaus_expected_stats, cat_expected_stats, lds_logZ):
    lds_params_p, hmm_params_p = prior_params
    lds_params, hmm_params = inference_params

    # cat kl
    cat_natparam = jumping_lds_to_hmm_mf(gaus_expected_stats, *lds_params)
    hmm_logZ = trans_hmm_inference(*tree_map(lambda x,y: x+y, cat_natparam, hmm_params))[1]
    cat_kl = trans_hmm_kl_full(cat_natparam, cat_expected_stats, hmm_logZ, hmm_params_p, hmm_params)

    # gaus_kl
    gaus_natparam, _ = jumping_hmm_to_lds_mf(cat_expected_stats, *lds_params)

    gaus_natparam_p, E_prior_logZ = jumping_hmm_to_lds_mf(cat_expected_stats, *lds_params_p)

    prior_params_lds = lds_transition_params_to_nat(*gaus_natparam_p)
    inference_params_lds = lds_transition_params_to_nat(*gaus_natparam)
    gaus_kl = lds_kl_full(recog_potentials, gaus_expected_stats,
                          *prior_params_lds, *inference_params_lds, E_prior_logZ, lds_logZ)

    return gaus_kl + cat_kl

@jit
def sample_jslds_stable(cat_expected_stats, recog_potentials, gaus_params, cat_params, key):
    gaus_natparam, _ = jumping_hmm_to_lds_mf(cat_expected_stats, *gaus_params)
    _, logZ, z = lds_inference_and_sample(recog_potentials, *gaus_natparam, key)
    return z.squeeze(-1), logZ

### Base function for implicit implementations

def jslds_inference_implicit_fwd(recog_potentials, lds_params, hmm_params, initializer):
    if initializer.shape == (2,):
        N, K = recog_potentials[0].shape[0], lds_params[0][0].shape[0]
        marginals = dirichlet(initializer, jnp.ones(K)*0.1, shape=(N,))
#         cat_expected_stats = jnp.ones((N,K))/K
    else:
        marginals = initializer

    init_es = jnp.expand_dims(marginals[0],-1).astype(recog_potentials[0].dtype)
    trans_es = vmap(jnp.outer)(marginals[:-1], marginals[1:]).astype(recog_potentials[0].dtype)
    cat_expected_stats = (init_es, trans_es)

    def cond_fun(vals):
        i, kl, old_kl, _, _ = vals
        return jnp.logical_and(abs(kl - old_kl)/old_kl >= 1e-5, i < MAX_ITER)

    def body_fun(vals):
        i, kl, _, _, cat_expected_stats = vals
        gaus_natparam, E_prior_logZ = jumping_hmm_to_lds_mf(cat_expected_stats, *lds_params)
        gaus_expected_stats, gaus_logZ, _ = lds_inference(recog_potentials, *gaus_natparam)
        gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, gaus_logZ)

        cat_natparam = jumping_lds_to_hmm_mf(gaus_expected_stats, *lds_params)
        cat_expected_stats, hmm_logZ = trans_hmm_inference(*tree_map(lambda x,y: x+y, cat_natparam, hmm_params))
        cat_kl = trans_hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
        return i+1, gaus_kl + cat_kl, kl, gaus_expected_stats, cat_expected_stats

    all_vals = body_fun((0, 1e10, jnp.inf, None, cat_expected_stats))
    i, kl, old_kl, gaus_expected_stats, cat_expected_stats = lax.while_loop(cond_fun, body_fun, all_vals)
#     id_print(i)

    host_callback.id_tap(lambda arg, _: wandb_log_internal(dict(coordinate_ascent_tol=jnp.max(jnp.abs(arg[0])), nconverged=jnp.mean(arg[1]))), ((kl - old_kl)/old_kl, i < MAX_ITER))
    all_args = recog_potentials, lds_params, hmm_params, gaus_expected_stats, cat_expected_stats, i

    return (gaus_expected_stats, cat_expected_stats), all_args

@jit
def parallel_update_global_vjp(recog_potentials, lds_params, hmm_params, gaus_expected_stats, cat_expected_stats):
    def parallel_update(recog_potentials, lds_params, hmm_params):
        gaus_natparam, _ = jumping_hmm_to_lds_mf(cat_expected_stats, *lds_params)
        cat_natparam = jumping_lds_to_hmm_mf(gaus_expected_stats, *lds_params)

        new_cat_es, _ = trans_hmm_inference(*tree_map(lambda x,y: x+y, cat_natparam, hmm_params))
        new_gaus_es, _, _ = lds_inference(recog_potentials, *gaus_natparam)
        return (new_gaus_es, new_cat_es)
    return vjp(parallel_update, recog_potentials, lds_params, hmm_params)

args_vmap = (0, None, None, 0)

### No solve

jslds_inference_nosolve = custom_vjp(jslds_inference_implicit)

def jslds_inference_nosolve_bwd(resids, grads):
    vjp_fun = parallel_update_global_vjp(*resids[:-1])[1]

    return vjp_fun(grads)

jslds_inference_nosolve.defvjp(jslds_inference_implicit_fwd, jslds_inference_nosolve_bwd)

### iterative solve

jslds_inference_itersolve = custom_vjp(jslds_inference_implicit)

@jit
def parallel_update_local_vjp(recog_potentials, lds_params, hmm_params, gaus_expected_stats, cat_expected_stats):
    def parallel_update(gaus_expected_stats, cat_expected_stats):
        gaus_natparam, _ = jumping_hmm_to_lds_mf(cat_expected_stats, *lds_params)
        cat_natparam = jumping_lds_to_hmm_mf(gaus_expected_stats, *lds_params)

        new_cat_es, _ = trans_hmm_inference(*tree_map(lambda x,y: x+y, cat_natparam, hmm_params))
        new_gaus_es, _, _ = lds_inference(recog_potentials, *gaus_natparam)
        return (new_gaus_es, new_cat_es)
    return vjp(parallel_update, gaus_expected_stats, cat_expected_stats)[1]

def jslds_inference_itersolve_bwd(resids, grads):
    with jax.default_matmul_precision('float32'):
        pu_args, maxiter = resids[:-1], resids[-1]
        square_vjp_fun = parallel_update_local_vjp(*pu_args)
        bwd_lr = 0.5

        def richardson_iter(arg):
            _, g, count = arg
            newg = square_vjp_fun(g)
    #         newg = tree_map(lambda x, y: x + y, newg, grads)
            newg = tree_map(lambda x, y, z: (1.-bwd_lr) * x + bwd_lr * y + bwd_lr * z, g, newg, grads)
            return g, newg, count + 1

        def richardson_cond(arg):
            prev, g, count = arg
            sse = tree_flatten(tree_map(lambda x, y: jnp.mean((x - y) ** 2), prev, g))[0]
            mse = sum(sse) / len(sse)
            return jnp.logical_and(mse > 1e-5, count < MAX_ITER)

        init = richardson_iter((grads, grads, 0))
        _, full_grads, count = lax.while_loop(richardson_cond, richardson_iter, init)

        # Log richardson residuals
        output = square_vjp_fun(full_grads)
        resid = jax.tree_map(lambda x,y,z: x+y-z, grads, output, full_grads)
        resid_rmse = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(resid)[0] ** 2))
        host_callback.id_tap(lambda resid, _: wandb_log_internal(dict(richardson_resid=jnp.max(resid))), resid_rmse)
    #     host_callback.id_tap(lambda resid, _: print(resid.shape), resid_rmse) # TODO remove

        cond = jnp.bitwise_or(resid_rmse > RICHARDSON_CLIPPING_THRESH, maxiter == MAX_ITER)
        full_grads = jax.tree_map(lambda a, b: jnp.where(cond, a, b), grads, full_grads)        
        host_callback.id_tap(lambda c, _: wandb_log_internal(dict(richardson_unconv=jnp.mean(c))), cond)

        vjp_fun = parallel_update_global_vjp(*pu_args)[1]

        return vjp_fun(full_grads) + (None,)

jslds_inference_itersolve.defvjp(jslds_inference_implicit_fwd, jslds_inference_itersolve_bwd)