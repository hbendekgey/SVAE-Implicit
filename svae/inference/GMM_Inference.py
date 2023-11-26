from jax import jit, vmap, jacrev, custom_vjp, vjp, tree_map, value_and_grad
from jax import lax, flatten_util
import jax
from functools import partial
from jax.tree_util import tree_flatten
from jax.scipy.linalg import solve
import jax.numpy as jnp
from inference.MP_Inference import lds_expected_stats_from_potentials, lds_to_hmm_mf, hmm_to_lds_mf, hmm_inference, lds_inference, hmm_kl, lds_kl, lds_transition_params_to_nat, hmm_kl_full, lds_kl_full, lds_to_hmm_mf_1step, gaus_to_cat_mf, single_gaus_kl, cat_to_gaus_mf, single_cat_kl
from distributions import normal, categorical
from utils import flatten_es, unflatten_es, make_csr, gaus_param_min
from jax.experimental import host_callback
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from utils import wandb_log, wandb_log_internal

MAX_ITER = 10
RICHARDSON_CLIPPING_THRESH = 1e2
CONV_THRESH = 1e-5

### Unrolled
def gmm_inference_unrolled_baseline(recog_potentials, gaus_global, gaus_normalizer, cat_global, initializer):
    gaus_expected_stats = normal.expected_stats_masked(recog_potentials)
    kl = jnp.inf
    for i in range(100):
        # Categorical Update
        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)
        cat_expected_stats = categorical.expected_stats(cat_natparam + cat_global)
        cat_kl = single_cat_kl(cat_expected_stats, cat_natparam, cat_global)

        gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        gaus_expected_stats = vmap(normal.expected_stats)(gaus_natparam)
        gaus_kl = single_gaus_kl(gaus_expected_stats, gaus_natparam, E_normalizer, recog_potentials)

        if abs(kl - (gaus_kl + cat_kl))/kl < CONV_THRESH:
            break
        kl = gaus_kl + cat_kl
#     print(i)
    return gaus_expected_stats, cat_expected_stats, kl

@jit
def gmm_inference_unrolled(recog_potentials, gaus_global, gaus_normalizer, cat_global, initializer):
    gaus_expected_stats = normal.expected_stats_masked(recog_potentials)
    gaus_expected_stats = tree_map(lambda x: x.astype(gaus_normalizer.dtype), gaus_expected_stats)

    def cond_fun(vals):
        kl, old_kl, _, _ = vals
        return abs(kl - old_kl)/old_kl >= CONV_THRESH

    def block_update(vals):
        kl, _, gaus_expected_stats, _ = vals
        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)
        cat_expected_stats = categorical.expected_stats(cat_natparam + cat_global)
        cat_kl = single_cat_kl(cat_expected_stats, cat_natparam, cat_global)

        gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        gaus_expected_stats = vmap(normal.expected_stats)(gaus_natparam)
        gaus_kl = single_gaus_kl(gaus_expected_stats, gaus_natparam, E_normalizer, recog_potentials)
        return jnp.ones((), dtype=gaus_normalizer.dtype) * (gaus_kl + cat_kl), jnp.ones((), dtype=gaus_normalizer.dtype) * kl, gaus_expected_stats, cat_expected_stats

    def body_fun(vals, _):
        return lax.cond(cond_fun(vals), block_update, lambda x: x, vals), 0

    all_vals = block_update((jnp.ones((), dtype=gaus_normalizer.dtype) * 1e10, jnp.ones((), dtype=gaus_normalizer.dtype) * jnp.inf, gaus_expected_stats, None))
    (kl, _, gaus_expected_stats, cat_expected_stats), _ = lax.scan(body_fun, all_vals, None, length=MAX_ITER-1)
#     id_print(i)
    return gaus_expected_stats, cat_expected_stats, kl

@jit
def gmm_kl(recog_potentials, gaus_global, gaus_normalizer, cat_global, gaus_expected_stats, cat_expected_stats):
    cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)
    cat_kl = single_cat_kl(cat_expected_stats, cat_natparam, cat_global)

    gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
    gaus_kl = single_gaus_kl(gaus_expected_stats, gaus_natparam, E_normalizer, recog_potentials)
    return gaus_kl + cat_kl

@jit
def gmm_kl_det(recog_potentials, prior_params, inference_params, gaus_expected_stats, cat_expected_stats):
    gaus_global_p, gaus_normalizer_p, cat_global_p = inference_params
    gaus_global, gaus_normalizer, cat_global = inference_params

    # cat kl
    cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)
    cat_kl = single_cat_kl_det(cat_expected_stats, cat_natparam, cat_global, cat_global_p)

    # gaus_kl
    gaus_natparam, _ = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
    gaus_natparam_p, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global_p, gaus_normalizer_p, recog_potentials)
    gaus_kl = single_gaus_kl_det(gaus_expected_stats, gaus_natparam, gaus_natparam_p, E_normalizer, recog_potentials)

    return gaus_kl + cat_kl

### Base function for implicit implementations

def gmm_inference_implicit(recog_potentials, gaus_global, gaus_normalizer, cat_global, initializer):
    gaus_expected_stats = normal.expected_stats_masked(recog_potentials)

    def cond_fun(vals):
        i, kl, old_kl, _, _ = vals
        return jnp.logical_and(abs(kl - old_kl)/old_kl >= CONV_THRESH, i < MAX_ITER)

    def body_fun(vals):
        i, kl, _, gaus_expected_stats, _ = vals
        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)
        cat_expected_stats = categorical.expected_stats(cat_natparam + cat_global)
        cat_kl = single_cat_kl(cat_expected_stats, cat_natparam, cat_global)

        gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        gaus_expected_stats = vmap(normal.expected_stats)(gaus_natparam)
        gaus_kl = single_gaus_kl(gaus_expected_stats, gaus_natparam, E_normalizer, recog_potentials)
        return i+1, gaus_kl + cat_kl, kl, gaus_expected_stats, cat_expected_stats

    all_vals = body_fun((0, 1e10, jnp.inf, gaus_expected_stats, None))
    i, _, _, gaus_expected_stats, cat_expected_stats = lax.while_loop(cond_fun, body_fun, all_vals)
#     id_print(i)
    return gaus_expected_stats, cat_expected_stats

def gmm_inference_implicit_fwd(recog_potentials, gaus_global, gaus_normalizer, cat_global, initializer):
    gaus_expected_stats = normal.expected_stats_masked(recog_potentials)

    def cond_fun(vals):
        i, kl, old_kl, _, _, _, _ = vals
        return jnp.logical_and(abs(kl - old_kl)/old_kl >= CONV_THRESH, i < MAX_ITER)

    def body_fun(vals):
        i, kl, _, gaus_expected_stats, _, _, _ = vals
        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)
        cat_expected_stats = categorical.expected_stats(cat_natparam + cat_global)
        cat_kl = single_cat_kl(cat_expected_stats, cat_natparam, cat_global)

        gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        gaus_expected_stats = vmap(normal.expected_stats)(gaus_natparam)
        gaus_kl = single_gaus_kl(gaus_expected_stats, gaus_natparam, E_normalizer, recog_potentials)
        return i+1, gaus_kl + cat_kl, kl, gaus_expected_stats, cat_expected_stats, gaus_natparam, cat_natparam

    all_vals = body_fun((0, 1e10, jnp.inf, gaus_expected_stats, None, None, None))
    i, _, _, gaus_expected_stats, cat_expected_stats, gaus_natparam, cat_natparam = lax.while_loop(cond_fun, body_fun, all_vals)
#     id_print(i)
    all_args = recog_potentials, gaus_global, gaus_normalizer, cat_global, gaus_expected_stats, cat_expected_stats, i

    return (gaus_expected_stats, cat_expected_stats), all_args

@jit
def parallel_update_global_vjp(recog_potentials, gaus_global, gaus_normalizer, cat_global,
                               gaus_expected_stats, cat_expected_stats):
    def parallel_update(recog_potentials, gaus_global, gaus_normalizer, cat_global):
        gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)

        new_gaus_es = vmap(normal.expected_stats)(gaus_natparam)
        new_cat_es = categorical.expected_stats(cat_natparam + cat_global)
        return (new_gaus_es, new_cat_es)
    return vjp(parallel_update, recog_potentials, gaus_global, gaus_normalizer, cat_global)

### Matrix solve implementation. 
# Obsolete (as it cannot be done inside jit) but preserved

gmm_inference_matsolve = custom_vjp(gmm_inference_implicit)

@vmap
def gmm_to_min(gaus_es, cat_es):
    es1 = gaus_es[0].flatten()
    es2 = gaus_es[1].flatten()
    return jnp.concatenate((es1, es2, cat_es))

@partial(vmap, in_axes=[0, None])
def gmm_from_min(flat_es, D):
    es1 = flat_es[:D**2].reshape(D,D)
    es2 = flat_es[D**2:D**2+D].reshape(D, 1)
    cat_es = flat_es[D**2+D:]
    return (es1, es2), cat_es

@partial(vmap, in_axes=[0, None, None, None, 0, None])
def parallel_update_jacobian(rp_flat, gaus_global, gaus_normalizer, cat_global, flat_es, D):
    recog_potentials = tree_map(lambda x: jnp.expand_dims(x,0), rp_flat)
    def parallel_update(flat_es):
        expanded_flat_es = jnp.expand_dims(flat_es,0)
        gaus_expected_stats, cat_expected_stats = gmm_from_min(expanded_flat_es, D)
        gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)

        new_gaus_es = vmap(normal.expected_stats)(gaus_natparam)
        new_cat_es = categorical.expected_stats(cat_natparam + cat_global)
        return gmm_to_min(new_gaus_es, new_cat_es)[0]
    return jacrev(parallel_update)(flat_es)

def gmm_inference_matsolve_bwd(resids, grads):
    flat_dim = grads[0][0].shape[-1]
    pu_args = resids[:-1]
    jac_args = (*resids[:-3], gmm_to_min(resids[-3], resids[-2]), flat_dim)

    J = parallel_update_jacobian(*jac_args)
    flat_grad = gmm_to_min(*grads)
    adjusted_grad = solve(jnp.identity(J.shape[-1]) - J.swapaxes(-1,-2), flat_grad)
    full_grad = gmm_from_min(adjusted_grad, flat_dim)

    vjp_fun = parallel_update_global_vjp(*resids[:-1])[1]
    return vjp_fun(full_grad)

gmm_inference_matsolve.defvjp(gmm_inference_implicit_fwd, gmm_inference_matsolve_bwd)

### No solve

gmm_inference_nosolve = custom_vjp(gmm_inference_implicit)

def gmm_inference_nosolve_bwd(resids, grads):
    vjp_fun = parallel_update_global_vjp(*resids[:-1])[1]
    return vjp_fun(grads)

gmm_inference_nosolve.defvjp(gmm_inference_implicit_fwd, gmm_inference_nosolve_bwd)

### iterative solve

gmm_inference_itersolve = custom_vjp(gmm_inference_implicit)

@jit
def parallel_update_local_vjp(recog_potentials, gaus_global, gaus_normalizer, cat_global,
                               gaus_expected_stats, cat_expected_stats):
    def parallel_update(gaus_expected_stats, cat_expected_stats):
        gaus_natparam, E_normalizer = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)

        new_gaus_es = vmap(normal.expected_stats)(gaus_natparam)
        new_cat_es = categorical.expected_stats(cat_natparam + cat_global)
        return (new_gaus_es, new_cat_es)
    return vjp(parallel_update, gaus_expected_stats, cat_expected_stats)[1]

def gmm_inference_itersolve_bwd(resids, grads):
    with jax.default_matmul_precision('float32'):
        pu_args, maxiter = resids[:-1], resids[-1]
        square_vjp_fun = parallel_update_local_vjp(*pu_args)
        bwd_lr = 0.5

        def richardson_iter(arg):
            _, g, count = arg
            newg = square_vjp_fun(g)
#             newg = tree_map(lambda x, y: x + y, newg, grads)
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
        resid = tree_map(lambda x,y,z: x+y-z, grads, output, full_grads)
        resid_rmse = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(resid)[0] ** 2))
        host_callback.id_tap(lambda resid, _: wandb_log_internal(dict(richardson_resid=jnp.max(resid))), resid_rmse)
    #     host_callback.id_tap(lambda resid, _: print(resid.shape), resid_rmse) # TODO remove

        cond = jnp.bitwise_or((resid_rmse > RICHARDSON_CLIPPING_THRESH) | jnp.isnan(resid_rmse), maxiter == MAX_ITER)
        full_grads = jax.tree_map(lambda a, b: jnp.where(cond, a, b), grads, full_grads)        
        host_callback.id_tap(lambda c, _: wandb_log_internal(dict(richardson_unconv=jnp.mean(c))), cond)

        vjp_fun = parallel_update_global_vjp(*pu_args)[1]
        return vjp_fun(full_grads) + (None,)

gmm_inference_itersolve.defvjp(gmm_inference_implicit_fwd, gmm_inference_itersolve_bwd)

### CGSolve

from jax.scipy.sparse.linalg import bicgstab, cg, gmres

gmm_inference_cgsolve = custom_vjp(gmm_inference_implicit)

def gmm_inference_cgsolve_bwd(resids, grads):
    pu_args, maxiter = resids[:-1], resids[-1]

    solver = partial(bicgstab, maxiter=maxiter)
    square_vjp_fun = parallel_update_local_vjp(*pu_args)
    one_step = lambda x: tree_map(lambda y,z: y - z, x, square_vjp_fun(x))
    full_grads = solver(one_step, grads)[0]
    vjp_fun = parallel_update_global_vjp(*pu_args)[1]
    return vjp_fun(full_grads)

gmm_inference_cgsolve.defvjp(gmm_inference_implicit_fwd, gmm_inference_cgsolve_bwd)