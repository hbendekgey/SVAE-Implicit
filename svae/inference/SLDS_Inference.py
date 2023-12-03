from jax import jit, vmap, jacrev, custom_vjp, vjp, tree_map, value_and_grad
from jax import lax, flatten_util
from functools import partial
from jax.tree_util import tree_flatten
import jax.numpy as jnp
from inference.MP_Inference import lds_expected_stats_from_potentials, lds_to_hmm_mf, hmm_to_lds_mf, hmm_inference, hsmm_inference, lds_inference, hmm_kl, lds_kl, lds_kl_surr, lds_transition_params_to_nat, hmm_kl_full, lds_kl_full, lds_kl_gen, hmm_kl_gen, lds_inference_and_sample
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

MAX_ITER = 10
RICHARDSON_CLIPPING_THRESH = 1e2
CONV_THRESH = 1e-5

def jit(x, *args, **kwargs):
    return x

### Unrolled
def slds_inference_unrolled_baseline(recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps, initializer):
    if initializer.shape == (2,):
        N, K = recog_potentials[0].shape[0] - 1, E_mniw_params[0].shape[0]
        cat_expected_stats = dirichlet(initializer, jnp.ones(K)*0.1, shape=(N,))
#         cat_expected_stats = jnp.ones((N,K))/K
    else:
        cat_expected_stats = initializer

    cat_expected_stats = cat_expected_stats.astype(E_init_lps.dtype)

    kl = jnp.inf
    for i in range(100):
        # Categorical Update
        gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        gaus_expected_stats, gaus_logZ, _ = lds_inference(recog_potentials, init, gaus_natparam)
        gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, gaus_logZ)

        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)
        cat_expected_stats, hmm_logZ, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
        cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)

        if abs(kl - (gaus_kl + cat_kl))/kl < CONV_THRESH:
            break
        kl = gaus_kl + cat_kl
#     print(i)
    return gaus_expected_stats, cat_expected_stats, kl

@jit
def slds_inference_unrolled(recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps, initializer):
    if initializer.shape == (2,):
        N, K = recog_potentials[0].shape[0] - 1, E_mniw_params[0].shape[0]
        cat_expected_stats = dirichlet(initializer, jnp.ones(K)*0.1, shape=(N,))
#         cat_expected_stats = jnp.ones((N,K))/K
    else:
        cat_expected_stats = initializer

    cat_expected_stats = cat_expected_stats.astype(E_init_lps.dtype)

    def cond_fun(vals):
        kl, old_kl, _, _ = vals
        return abs(kl - old_kl)/old_kl >= CONV_THRESH

    def block_update(vals):
        kl, _, _, cat_expected_stats = vals
        gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        gaus_expected_stats, gaus_logZ, lds_messages = lds_inference(recog_potentials, init, gaus_natparam)
        gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, gaus_logZ)

        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)
        cat_expected_stats, hmm_logZ, hmm_messages = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
        cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
        return jnp.ones((), dtype=E_init_lps.dtype) * (gaus_kl + cat_kl), jnp.ones((), dtype=E_init_lps.dtype) * kl, gaus_expected_stats, cat_expected_stats

    def body_fun(vals, _):
        return lax.cond(cond_fun(vals), block_update, lambda x: x, vals), 0

    all_vals = block_update((jnp.ones((), dtype=E_init_lps.dtype) * 1e10, jnp.ones((), dtype=E_init_lps.dtype) * jnp.inf, None, cat_expected_stats))
    (kl, _, gaus_expected_stats, cat_expected_stats), _ = lax.scan(body_fun, all_vals, None, length=MAX_ITER-1)
#     id_print(i)
    return gaus_expected_stats, cat_expected_stats

args_vmap = (0, None, None, None, None, None, 0)

slds_inference_unrolled_batched = vmap(slds_inference_unrolled, in_axes=args_vmap)

@jit
def slds_kl(recog_potentials, E_mniw_params, init, E_init_normalizer,
            E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats, lds_logZ):
    gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
    cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

    new_cat_es, hmm_logZ, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
#     new_gaus_es, lds_logZ, _ = lds_inference(recog_potentials, init, gaus_natparam)
    gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, lds_logZ)
    cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
    return gaus_kl + cat_kl

@jit
def slds_kl_sur(recog_potentials, E_mniw_params, init, E_init_normalizer,
            E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats, lds_logZ):
    gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
    cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

    _, hmm_logZ, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
#     new_gaus_es, lds_logZ, _ = lds_inference(recog_potentials, init, gaus_natparam)
    gaus_kl = lds_kl_surr(recog_potentials, gaus_expected_stats, E_prior_logZ, lds_logZ)
    cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
    return gaus_kl + cat_kl

@jit
def slds_kl_det(recog_potentials, prior_params, inference_params, gaus_expected_stats, cat_expected_stats, lds_logZ):
    E_mniw_params_p, init_p, E_init_normalizer_p, E_init_lps_p, E_trans_lps_p = prior_params
    E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps = inference_params

    # cat kl
    cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)
    def cat_es_full(E_trans_lps):
        return hmm_inference(E_init_lps, E_trans_lps, cat_natparam)[1]
    hmm_logZ, EZZNT = value_and_grad(cat_es_full)(E_trans_lps)
    cat_kl = hmm_kl_full(cat_natparam, cat_expected_stats, hmm_logZ, E_init_lps_p, E_trans_lps_p,
                         E_init_lps, E_trans_lps, EZZNT)

    # gaus_kl
    gaus_natparam, _ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
#     lds_logZ = lds_inference(recog_potentials, init, gaus_natparam)[1]

    gaus_natparam_p, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params_p, E_init_normalizer_p)

    prior_params_lds = lds_transition_params_to_nat(init_p, gaus_natparam_p)
    inference_params_lds = lds_transition_params_to_nat(init, gaus_natparam)
    gaus_kl = lds_kl_full(recog_potentials, gaus_expected_stats,
                          *prior_params_lds, *inference_params_lds, E_prior_logZ, lds_logZ)

    return gaus_kl + cat_kl

@jit
def slds_crossE(recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps, 
                gaus_expected_stats, cat_expected_stats, lds_logZ):
    gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
    cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

    init, gaus_natparam = lds_transition_params_to_nat(init, gaus_natparam)
    gaus_kl = lds_kl_gen(gaus_expected_stats, init, gaus_natparam, E_prior_logZ)
    
    def cat_es_full(E_trans_lps):
        return hmm_inference(E_init_lps, E_trans_lps, cat_natparam)[1]
    _, EZZNT = value_and_grad(cat_es_full)(E_trans_lps)
    cat_kl = hmm_kl_gen(cat_expected_stats, E_init_lps, E_trans_lps, EZZNT)    
    return gaus_kl + cat_kl

# begin lots of code to see every part of the surrogate
@jit
def slds_kl_full(recog_potentials, E_mniw_params, init, E_init_normalizer,
            E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats, lds_logZ):
    gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
    cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

    new_cat_es, hmm_logZ, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
#     new_gaus_es, lds_logZ, _ = lds_inference(recog_potentials, init, gaus_natparam)
    gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, lds_logZ)
    cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
    return gaus_kl, cat_kl

@jit
def slds_kl_sur_full(recog_potentials, E_mniw_params, init, E_init_normalizer,
            E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats, lds_logZ):
    gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
    cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

    new_cat_es, hmm_logZ, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
#     new_gaus_es, lds_logZ, _ = lds_inference(recog_potentials, init, gaus_natparam)
    gaus_kl = lds_kl_surr(recog_potentials, gaus_expected_stats, E_prior_logZ, lds_logZ)
    cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
    return gaus_kl, cat_kl

@jit
def slds_crossE_full(recog_potentials, E_mniw_params, init, E_init_normalizer,
            E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats, lds_logZ):
    gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
    cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

    init, gaus_natparam = lds_transition_params_to_nat(init, gaus_natparam)
    gaus_kl = lds_kl_gen(gaus_expected_stats, init, gaus_natparam, E_prior_logZ)
    
    def cat_es_full(E_trans_lps):
        return hmm_inference(E_init_lps, E_trans_lps, cat_natparam)[1]
    _, EZZNT = value_and_grad(cat_es_full)(E_trans_lps)
    cat_kl = hmm_kl_gen(cat_expected_stats, E_init_lps, E_trans_lps, EZZNT)    
    return gaus_kl, cat_kl

@jit
def slds_surr_decomp(*args):
    gaus_kl, cat_kl = slds_kl_full(*args)
    gaus_kl_minus_recog, _ = slds_kl_sur_full(*args)
    negElogpz, negElogpk = slds_crossE_full(*args)
    return gaus_kl - negElogpz, negElogpz, cat_kl - negElogpk, negElogpk, gaus_kl_minus_recog - gaus_kl


@jit
def sample_slds_stable(cat_expected_stats, recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps, key):
    gaus_natparam, _ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
    _, logZ, z = lds_inference_and_sample(recog_potentials, init, gaus_natparam, key)
    return z.squeeze(-1), logZ

### Base function for implicit implementations

def slds_inference_implicit(recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps, initializer):
    if initializer.shape == (2,):
        N, K = recog_potentials[0].shape[0] - 1, E_mniw_params[0].shape[0]
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
        gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        gaus_expected_stats, gaus_logZ, lds_messages = lds_inference(recog_potentials, init, gaus_natparam)
        gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, gaus_logZ)

        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)
        cat_expected_stats, hmm_logZ, hmm_messages = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
        cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
        return i+1, gaus_kl + cat_kl, kl, gaus_expected_stats, cat_expected_stats

    all_vals = body_fun((0, 1e10, jnp.inf, None, cat_expected_stats))
    i, _, _, gaus_expected_stats, cat_expected_stats = lax.while_loop(cond_fun, body_fun, all_vals)
#     id_print(i)
    return gaus_expected_stats, cat_expected_stats

def slds_inference_implicit_fwd(recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps, initializer):
    if initializer.shape == (2,):
        N, K = recog_potentials[0].shape[0] - 1, E_mniw_params[0].shape[0]
        cat_expected_stats = dirichlet(initializer, jnp.ones(K)*0.1, shape=(N,))
#         cat_expected_stats = jnp.ones((N,K))/K
    else:
        cat_expected_stats = initializer

    cat_expected_stats = cat_expected_stats.astype(recog_potentials[0].dtype)

    def cond_fun(vals):
        i, kl, old_kl, _, _, _, _, _ = vals
        return jnp.logical_and(abs(kl - old_kl)/old_kl >= CONV_THRESH, i < MAX_ITER)

    def body_fun(vals):
        i, kl, _, _, cat_expected_stats, _, _, _ = vals
        gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        gaus_expected_stats, gaus_logZ, lds_messages = lds_inference(recog_potentials, init, gaus_natparam)
        gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, gaus_logZ)

        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)
        cat_expected_stats, hmm_logZ, hmm_messages = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
        cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)

        return i+1, gaus_kl + cat_kl, kl, gaus_expected_stats, cat_expected_stats, gaus_natparam, cat_natparam, (lds_messages, hmm_messages)

    all_vals = body_fun((0, 1e10, jnp.inf, None, cat_expected_stats, None, None, None))
    i, kl, old_kl, gaus_expected_stats, cat_expected_stats, gaus_natparam, cat_natparam, messages = lax.while_loop(cond_fun, body_fun, all_vals)
#     id_print(i)

    # Log convergence metrics
    # host_callback.id_tap(lambda arg, _: print(('Shapes:', arg[0].shape, arg[1].shape)), (kl - old_kl, i < MAX_ITER))
    host_callback.id_tap(lambda arg, _: wandb_log_internal(dict(coordinate_ascent_tol=jnp.max(jnp.abs(arg[0])), nconverged=jnp.mean(arg[1]))), ((kl - old_kl)/old_kl, i < MAX_ITER))
    all_args = recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats, gaus_natparam, cat_natparam, messages, i

    return (gaus_expected_stats, cat_expected_stats), all_args

@jit
def parallel_update_global_vjp(recog_potentials, E_mniw_params, init, E_init_normalizer,
                               E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats):
    def parallel_update(recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps):
        gaus_natparam, _ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

        new_cat_es, _, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
        new_gaus_es, _, _ = lds_inference(recog_potentials, init, gaus_natparam)
        return (new_gaus_es, new_cat_es)
    return vjp(parallel_update, recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps)

args_vmap = (0, None, None, None, None, None, 0)

slds_inference_implicit_batched_fwd = vmap(slds_inference_implicit_fwd, 
                                           in_axes=args_vmap, out_axes=(0, args_vmap + (0,0,0,0,0)))

### No solve

slds_inference_nosolve = custom_vjp(slds_inference_implicit)

def slds_inference_nosolve_bwd(resids, grads):
    vjp_fun = parallel_update_global_vjp(*resids[:-4])[1]

    return vjp_fun(grads) + (None,)

slds_inference_nosolve.defvjp(slds_inference_implicit_fwd, slds_inference_nosolve_bwd)

### And Batch it

slds_inference_nosolve_batched = custom_vjp(vmap(slds_inference_implicit, in_axes=args_vmap))

def slds_inference_nosolve_batched_bwd(resids, grads):
    combined_grads = vmap(slds_inference_nosolve_bwd, in_axes=(args_vmap + (0,0,0,0,0), 0))(resids, grads)
    return (combined_grads[0],) + tree_map(lambda x: x.sum(0), combined_grads[1:-1]) + (None,)

slds_inference_nosolve_batched.defvjp(slds_inference_implicit_batched_fwd, slds_inference_nosolve_batched_bwd)

### iterative solve

slds_inference_itersolve = custom_vjp(slds_inference_implicit)

@jit
def parallel_update_local_vjp(recog_potentials, E_mniw_params, init, E_init_normalizer,
                              E_init_lps, E_trans_lps, gaus_expected_stats, cat_expected_stats):
    def parallel_update(gaus_expected_stats, cat_expected_stats):
        gaus_natparam, _ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

        new_cat_es, _, _ = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)
        new_gaus_es, _, _ = lds_inference(recog_potentials, init, gaus_natparam)
        return (new_gaus_es, new_cat_es)
    return vjp(parallel_update, gaus_expected_stats, cat_expected_stats)[1]

def slds_inference_itersolve_bwd(resids, grads):
    with jax.default_matmul_precision('float32'):
        pu_args = resids[:-4]
        maxiter = resids[-1]
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
        host_callback.id_tap(lambda resid, _: wandb_log_internal(dict(richardson_resid=jnp.max(resid))), resid_rmse)
    #     host_callback.id_tap(lambda resid, _: print(resid.shape), resid_rmse) # TODO remove

        cond = jnp.bitwise_or((resid_rmse > RICHARDSON_CLIPPING_THRESH) | jnp.isnan(resid_rmse), maxiter == MAX_ITER)
        full_grads = jax.tree_map(lambda a, b: jnp.where(cond, a, b), grads, full_grads)        
        host_callback.id_tap(lambda c, _: wandb_log_internal(dict(richardson_unconv=jnp.mean(c))), cond)

        vjp_fun = parallel_update_global_vjp(*pu_args)[1]
        return vjp_fun(full_grads) + (None,)

slds_inference_itersolve.defvjp(slds_inference_implicit_fwd, slds_inference_itersolve_bwd)

slds_inference_itersolve_uncapped = custom_vjp(slds_inference_implicit)

def slds_inference_itersolve_uncapped_bwd(resids, grads):
    pu_args = resids[:-4]
    maxiter = resids[-1]
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
        return jnp.logical_and(mse > 0, count < 100)

    init = richardson_iter((grads, grads, 0))
    _, full_grads, count = lax.while_loop(richardson_cond, richardson_iter, init)
#     id_print(count)

    vjp_fun = parallel_update_global_vjp(*pu_args)[1]

    return vjp_fun(full_grads) + (None,)

slds_inference_itersolve_uncapped.defvjp(slds_inference_implicit_fwd, slds_inference_itersolve_uncapped_bwd)

### And Batch it

slds_inference_itersolve_batched = custom_vjp(vmap(slds_inference_implicit, in_axes=args_vmap))

def slds_inference_itersolve_batched_bwd(resids, grads):
    combined_grads = vmap(slds_inference_itersolve_bwd, in_axes=(args_vmap + (0,0,0,0,0), 0))(resids, grads)
    return (combined_grads[0],) + tree_map(lambda x: x.sum(0), combined_grads[1:-1]) + (None,)

slds_inference_itersolve_batched.defvjp(slds_inference_implicit_batched_fwd, slds_inference_itersolve_batched_bwd)

slds_inference_itersolve_uncapped_batched = custom_vjp(vmap(slds_inference_implicit, in_axes=args_vmap))

def slds_inference_itersolve_uncapped_batched_bwd(resids, grads):
    combined_grads = vmap(slds_inference_itersolve_uncapped_bwd, in_axes=(args_vmap + (0,0,0,0,0), 0))(resids, grads)
    return (combined_grads[0],) + tree_map(lambda x: x.sum(0), combined_grads[1:-1]) + (None,)

slds_inference_itersolve_uncapped_batched.defvjp(slds_inference_implicit_batched_fwd, slds_inference_itersolve_uncapped_batched_bwd)


### CGSolve

from jax.scipy.sparse.linalg import bicgstab, cg, gmres

slds_inference_cgsolve = custom_vjp(slds_inference_implicit)

def slds_inference_cgsolve_bwd(resids, grads):
    pu_args = resids[:-4]
    maxiter = resids[-1]
    solver = partial(bicgstab, maxiter=maxiter)

    square_vjp_fun = parallel_update_local_vjp(*pu_args)
    one_step = lambda x: tree_map(lambda y,z: y - z, x, square_vjp_fun(x))
    full_grads = solver(one_step, grads)[0]
    vjp_fun = parallel_update_global_vjp(*pu_args)[1]
    return vjp_fun(full_grads) + (None,)

slds_inference_cgsolve.defvjp(slds_inference_implicit_fwd, slds_inference_cgsolve_bwd)

### And Batch it

slds_inference_cgsolve_batched = custom_vjp(vmap(slds_inference_implicit, in_axes=args_vmap))

def slds_inference_cgsolve_batched_bwd(resids, grads):
    combined_grads = vmap(slds_inference_cgsolve_bwd, in_axes=(args_vmap + (0,0,0,0,0), 0))(resids, grads)
    return (combined_grads[0],) + tree_map(lambda x: x.sum(0), combined_grads[1:-1]) + (None,)

slds_inference_cgsolve_batched.defvjp(slds_inference_implicit_batched_fwd, slds_inference_cgsolve_batched_bwd)







def sm_slds_inference_implicit(recog_potentials, E_mniw_params, init, E_init_normalizer, 
                               E_init_lps, E_trans_lps, E_self_trans_lps, initializer):
    if initializer.shape == (2,):
        N, K = recog_potentials[0].shape[0] - 1, E_mniw_params[0].shape[0]
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
        gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        gaus_expected_stats, gaus_logZ, _ = lds_inference(recog_potentials, init, gaus_natparam)
        gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, gaus_logZ)

        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)
        cat_expected_stats, hmm_logZ, _ = hsmm_inference(E_init_lps, E_trans_lps, E_self_trans_lps, cat_natparam)
        cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
        return i+1, gaus_kl + cat_kl, kl, gaus_expected_stats, cat_expected_stats

    all_vals = body_fun((0, 1e10, jnp.inf, None, cat_expected_stats))
    i, _, _, gaus_expected_stats, cat_expected_stats = lax.while_loop(cond_fun, body_fun, all_vals)
#     id_print(i)
    return gaus_expected_stats, cat_expected_stats

def sm_slds_inference_implicit_fwd(recog_potentials, E_mniw_params, init, E_init_normalizer, 
                                   E_init_lps, E_trans_lps, E_self_trans_lps, initializer):
    if initializer.shape == (2,):
        N, K = recog_potentials[0].shape[0] - 1, E_mniw_params[0].shape[0]
        cat_expected_stats = dirichlet(initializer, jnp.ones(K)*0.1, shape=(N,))
#         cat_expected_stats = jnp.ones((N,K))/K
    else:
        cat_expected_stats = initializer

    cat_expected_stats = cat_expected_stats.astype(recog_potentials[0].dtype)

    def cond_fun(vals):
        i, kl, old_kl, _, _, _, _ = vals
        return jnp.logical_and(abs(kl - old_kl)/old_kl >= CONV_THRESH, i < MAX_ITER)

    def body_fun(vals):
        i, kl, _, _, cat_expected_stats, _, _ = vals
        gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        gaus_expected_stats, gaus_logZ, _ = lds_inference(recog_potentials, init, gaus_natparam)
        gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, gaus_logZ)

        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)
        cat_expected_stats, hmm_logZ, _ = hsmm_inference(E_init_lps, E_trans_lps, E_self_trans_lps, cat_natparam)
        cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)

        return i+1, gaus_kl + cat_kl, kl, gaus_expected_stats, cat_expected_stats, gaus_natparam, cat_natparam

    all_vals = body_fun((0, 1e10, jnp.inf, None, cat_expected_stats, None, None))
    i, kl, old_kl, gaus_expected_stats, cat_expected_stats, gaus_natparam, cat_natparam = lax.while_loop(cond_fun, body_fun, all_vals)
#     id_print(i)

    # Log convergence metrics
    # host_callback.id_tap(lambda arg, _: print(('Shapes:', arg[0].shape, arg[1].shape)), (kl - old_kl, i < MAX_ITER))
    host_callback.id_tap(lambda arg, _: wandb_log_internal(dict(coordinate_ascent_tol=jnp.max(jnp.abs(arg[0])), nconverged=jnp.mean(arg[1]))), ((kl - old_kl)/old_kl, i < MAX_ITER))
    all_args = recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps, E_self_trans_lps, gaus_expected_stats, cat_expected_stats, i

    return (gaus_expected_stats, cat_expected_stats), all_args

sm_slds_inference_itersolve = custom_vjp(sm_slds_inference_implicit)

@jit
def sm_parallel_update_local_vjp(recog_potentials, E_mniw_params, init, E_init_normalizer,
                                 E_init_lps, E_trans_lps, E_self_trans_lps, gaus_expected_stats, cat_expected_stats):
    def parallel_update(gaus_expected_stats, cat_expected_stats):
        gaus_natparam, _ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

        new_cat_es, _, _ = hsmm_inference(E_init_lps, E_trans_lps, E_self_trans_lps, cat_natparam)
        new_gaus_es, _, _ = lds_inference(recog_potentials, init, gaus_natparam)
        return (new_gaus_es, new_cat_es)
    return vjp(parallel_update, gaus_expected_stats, cat_expected_stats)[1]

@jit
def sm_parallel_update_global_vjp(recog_potentials, E_mniw_params, init, E_init_normalizer,
                                  E_init_lps, E_trans_lps, E_self_trans_lps, gaus_expected_stats, cat_expected_stats):
    def parallel_update(recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps, E_self_trans_lps):
        gaus_natparam, _ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

        new_cat_es, _, _ = hsmm_inference(E_init_lps, E_trans_lps, E_self_trans_lps, cat_natparam)
        new_gaus_es, _, _ = lds_inference(recog_potentials, init, gaus_natparam)
        return (new_gaus_es, new_cat_es)
    return vjp(parallel_update, recog_potentials, E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps, E_self_trans_lps)

def sm_slds_inference_itersolve_bwd(resids, grads):
    with jax.default_matmul_precision('float32'):
        pu_args = resids[:-1]
        maxiter = resids[-1]
        square_vjp_fun = sm_parallel_update_local_vjp(*pu_args)
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
        host_callback.id_tap(lambda resid, _: wandb_log_internal(dict(richardson_resid=jnp.max(resid))), resid_rmse)
    #     host_callback.id_tap(lambda resid, _: print(resid.shape), resid_rmse) # TODO remove

        cond = jnp.bitwise_or((resid_rmse > RICHARDSON_CLIPPING_THRESH) | jnp.isnan(resid_rmse), maxiter == MAX_ITER)
        full_grads = jax.tree_map(lambda a, b: jnp.where(cond, a, b), grads, full_grads)        
        host_callback.id_tap(lambda c, _: wandb_log_internal(dict(richardson_unconv=jnp.mean(c))), cond)

        vjp_fun = sm_parallel_update_global_vjp(*pu_args)[1]


        return vjp_fun(full_grads) + (None,)

sm_slds_inference_itersolve.defvjp(sm_slds_inference_implicit_fwd, sm_slds_inference_itersolve_bwd)

@jit
def sm_slds_kl(recog_potentials, E_mniw_params, init, E_init_normalizer,
            E_init_lps, E_trans_lps, E_self_trans_lps, gaus_expected_stats, cat_expected_stats, lds_logZ):
    gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
    cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

    new_cat_es, hmm_logZ, _ = hsmm_inference(E_init_lps, E_trans_lps, E_self_trans_lps, cat_natparam)
#     new_gaus_es, lds_logZ, _ = lds_inference(recog_potentials, init, gaus_natparam)
    gaus_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, lds_logZ)
    cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
    return gaus_kl + cat_kl

@jit
def sm_slds_kl_sur(recog_potentials, E_mniw_params, init, E_init_normalizer,
            E_init_lps, E_trans_lps, E_self_trans_lps, gaus_expected_stats, cat_expected_stats, lds_logZ):
    gaus_natparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
    cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)

    _, hmm_logZ, _ = hsmm_inference(E_init_lps, E_trans_lps, E_self_trans_lps, cat_natparam)
#     new_gaus_es, lds_logZ, _ = lds_inference(recog_potentials, init, gaus_natparam)
    gaus_kl = lds_kl_surr(recog_potentials, gaus_expected_stats, E_prior_logZ, lds_logZ)
    cat_kl = hmm_kl(cat_natparam, cat_expected_stats, hmm_logZ)
    return gaus_kl + cat_kl
