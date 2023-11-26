import jax
from jax import jit, vmap, jacrev, custom_vjp, vjp, tree_map, value_and_grad
from jax import lax
from functools import partial
from jax.tree_util import tree_flatten
import jax.numpy as jnp
from inference.MP_Inference import hmm_inference, hmm_kl, hmm_kl_full, gaus_to_cat_mf, single_gaus_kl, cat_to_gaus_mf, single_gaus_kl_sur
from utils import flatten_es, unflatten_es, make_csr, gaus_param_min
from jax.experimental.host_callback import call, id_print, id_tap
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve
import time
import numpy as np
from distributions import normal
from jax.random import dirichlet, split
from jax.experimental import host_callback
from utils import wandb_log, wandb_log_internal


MAX_ITER = 10
RICHARDSON_CLIPPING_THRESH = 1e2
CONV_THRESH = 1e-5

def hmm_mf_inference_implicit(recog_potentials, gaus_global, gaus_normalizer, E_init_lps, E_trans_lps, initializer):
    if initializer.shape == (2,):
        N, K = recog_potentials[0].shape[0], E_trans_lps.shape[0]
        cat_expected_stats = dirichlet(initializer, jnp.ones(K)*0.1, shape=(N,))
#         cat_expected_stats = jnp.ones((N,K))/K
    else:
        cat_expected_stats = initializer

    cat_expected_stats = cat_expected_stats.astype(recog_potentials[0].dtype)

    def cond_fun(vals):
        i, kl, old_kl, _, _ = vals
        return jnp.logical_and(abs(kl - old_kl)/old_kl >= CONV_THRESH, i < MAX_ITER)

    def body_fun(vals):
        i, kl, _, _, cat_expected_stats = vals
        gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        gaus_expected_stats = vmap(normal.expected_stats)(gaus_natparam)
        gaus_kl = single_gaus_kl(gaus_expected_stats, gaus_natparam, E_normalizer, recog_potentials)

        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)
        cat_expected_stats, hmm_logZ, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
        cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
        return i+1, gaus_kl + cat_kl, kl, gaus_expected_stats, cat_expected_stats

    all_vals = body_fun((0, 1e10, jnp.inf, None, cat_expected_stats))
    i, _, _, gaus_expected_stats, cat_expected_stats = lax.while_loop(cond_fun, body_fun, all_vals)
#     id_print(i)
    return gaus_expected_stats, cat_expected_stats

@jit
def sample_hmm_mf(gaus_expected_stats, key):
    keys = split(key, gaus_expected_stats[0].shape[0])
    return vmap(normal.sample_from_es)(gaus_expected_stats, keys).squeeze(1)

@jit
def hmm_mf_kl(recog_potentials, gaus_global, gaus_normalizer, E_init_lps, 
            E_trans_lps, gaus_expected_stats, cat_expected_stats):
    gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
    cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)

    new_cat_es, hmm_logZ, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
    new_gaus_es = vmap(normal.expected_stats)(gaus_natparam)
    gaus_kl = single_gaus_kl(gaus_expected_stats, gaus_natparam, E_normalizer, recog_potentials)
    cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
    return gaus_kl + cat_kl

@jit
def hmm_mf_kl_sur(recog_potentials, gaus_global, gaus_normalizer, E_init_lps, 
            E_trans_lps, gaus_expected_stats, cat_expected_stats):
    gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
    cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)

    new_cat_es, hmm_logZ, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
    new_gaus_es = vmap(normal.expected_stats)(gaus_natparam)
    gaus_kl = single_gaus_kl_sur(gaus_expected_stats, gaus_natparam, E_normalizer, recog_potentials)
    cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
    return gaus_kl + cat_kl

@jit
def hmm_mf_kl_det(recog_potentials, prior_params, inference_params, gaus_expected_stats, cat_expected_stats):
    gaus_global_p, gaus_normalizer_p, E_init_lps_p, E_trans_lps_p = prior_params
    gaus_global, gaus_normalizer, E_init_lps, E_trans_lps = inference_params

    # cat kl
    cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)
    def cat_es_full(E_trans_lps):
        return hmm_inference(E_init_lps, E_trans_lps, cat_natparam)[1]
    hmm_logZ, EZZNT = value_and_grad(cat_es_full)(E_trans_lps)
    cat_kl = hmm_kl_full(cat_natparam, cat_expected_stats, hmm_logZ, E_init_lps_p, E_trans_lps_p,
                         E_init_lps, E_trans_lps, EZZNT)

    # gaus_kl
    gaus_natparam, _ = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
    gaus_natparam_p, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global_p, gaus_normalizer_p, recog_potentials)
    gaus_kl = single_gaus_kl_det(gaus_expected_stats, gaus_natparam, gaus_natparam_p, E_normalizer, recog_potentials)

    return gaus_kl + cat_kl

### Base function for implicit implementations

def hmm_mf_inference_implicit_fwd(recog_potentials, gaus_global, gaus_normalizer, 
                                  E_init_lps, E_trans_lps, initializer):
    if initializer.shape == (2,):
        N, K = recog_potentials[0].shape[0], E_trans_lps.shape[0]
        cat_expected_stats = dirichlet(initializer, jnp.ones(K)*0.1, shape=(N,))
#         cat_expected_stats = jnp.ones((N,K))/K
    else:
        cat_expected_stats = initializer

    cat_expected_stats = cat_expected_stats.astype(recog_potentials[0].dtype)

    def cond_fun(vals):
        i, kl, old_kl, _, _ = vals
        return jnp.logical_and(abs(kl - old_kl)/old_kl >= CONV_THRESH, i < MAX_ITER)

    def body_fun(vals):
        i, kl, _, _, cat_expected_stats = vals
        gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        gaus_expected_stats = vmap(normal.expected_stats)(gaus_natparam)
        gaus_kl = single_gaus_kl(gaus_expected_stats, gaus_natparam, E_normalizer, recog_potentials)

        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)
        cat_expected_stats, hmm_logZ, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
        cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
        return i+1, gaus_kl + cat_kl, kl, gaus_expected_stats, cat_expected_stats

    all_vals = body_fun((0, 1e10, jnp.inf, None, cat_expected_stats))
    i, kl, old_kl, gaus_expected_stats, cat_expected_stats = lax.while_loop(cond_fun, body_fun, all_vals)
#     id_print(i)

    host_callback.id_tap(lambda arg, _: wandb_log_internal(dict(coordinate_ascent_tol=np.array(jnp.max(jnp.abs(arg[0]))), nconverged=np.array(jnp.mean(arg[1])))), ((kl - old_kl)/old_kl, i < MAX_ITER))
    all_args = recog_potentials, gaus_global, gaus_normalizer, E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats, i
    return (gaus_expected_stats, cat_expected_stats), all_args

@jit
def parallel_update_global_vjp(recog_potentials, gaus_global, gaus_normalizer, 
                               E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats):
    def parallel_update(recog_potentials, gaus_global, gaus_normalizer, E_init_lps, E_trans_lps):
        gaus_natparam, _ = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)

        new_cat_es, _, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
        new_gaus_es = vmap(normal.expected_stats)(gaus_natparam)
        return (new_gaus_es, new_cat_es)
    return vjp(parallel_update, recog_potentials, gaus_global, gaus_normalizer, E_init_lps, E_trans_lps)

### iterative solve

hmm_mf_inference_itersolve = custom_vjp(hmm_mf_inference_implicit)

@jit
def parallel_update_local_vjp(recog_potentials, gaus_global, gaus_normalizer, 
                               E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats):
    def parallel_update(gaus_expected_stats, cat_expected_stats):
        gaus_natparam, _ = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)

        new_cat_es, _, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
        new_gaus_es = vmap(normal.expected_stats)(gaus_natparam)
        return (new_gaus_es, new_cat_es)

    return vjp(parallel_update, gaus_expected_stats, cat_expected_stats)[1]

def hmm_mf_inference_itersolve_bwd(resids, grads):
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
            return jnp.logical_and(mse > CONV_THRESH, count < MAX_ITER)

        init = richardson_iter((grads, grads, 0))
        _, full_grads, count = lax.while_loop(richardson_cond, richardson_iter, init)

        # Log richardson residuals
        output = square_vjp_fun(full_grads)
        resid = jax.tree_map(lambda x,y,z: x+y-z, grads, output, full_grads)
        resid_rmse = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(resid)[0] ** 2))
        host_callback.id_tap(lambda resid, _: wandb_log_internal(dict(richardson_resid=np.array(jnp.max(resid)))), resid_rmse)
    #     host_callback.id_tap(lambda resid, _: print(resid.shape), resid_rmse) # TODO remove

        cond = jnp.bitwise_or(resid_rmse > RICHARDSON_CLIPPING_THRESH, maxiter == MAX_ITER)
        full_grads = jax.tree_map(lambda a, b: jnp.where(cond, a, b), grads, full_grads)        
        host_callback.id_tap(lambda c, _: wandb_log_internal(dict(richardson_unconv=np.array(jnp.mean(c)))), cond)

        vjp_fun = parallel_update_global_vjp(*pu_args)[1]

        return vjp_fun(full_grads) + (None,)

hmm_mf_inference_itersolve.defvjp(hmm_mf_inference_implicit_fwd, hmm_mf_inference_itersolve_bwd)