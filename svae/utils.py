from jax import jit, vmap, tree_map, custom_vjp
from jax.lax import stop_gradient
from jax.numpy import swapaxes, pad, zeros, zeros_like, ones, arange, tile, diag_indices, concatenate, tril, triu, tril_indices, expand_dims, split, log, exp, where, identity, diag
from jax.numpy import sum as jnp_sum
from jax.numpy.linalg import cholesky
from jax.scipy.linalg import inv, solve, eigh 
from scipy.sparse import csr_matrix
import jax.numpy as jnp
import jax
from jax.nn import softplus
from functools import partial
from tensorflow_probability.substrates.jax.math import softplus_inverse 
from tensorflow_probability.substrates.jax import bijectors as tfb
import os
import wandb
import numpy as np
from jax.scipy.special import gammaln

def binom(x, y):
    return jnp.exp(gammaln(x + 1) - gammaln(y + 1) - gammaln(x - y + 1))

def wandb_log(info, state=None, step=None, **kwargs):
    if not (wandb.run is None):
        step = step or state.step
        if hasattr(wandb, 'aux') and type(wandb.aux) is dict:
            info = {**info, **wandb.aux}
            wandb.aux = None
        for k in info:
            info[k] = np.array(info[k])
        wandb.log(info, step=step, **kwargs)
    
def wandb_log_internal(info):
    if not (wandb.run is None):
        if not (hasattr(wandb, 'aux') and type(wandb.aux) is dict):
            wandb.aux = {}
        wandb.aux = {**info, **wandb.aux}
        
T = lambda x: x.swapaxes(-1,-2)

def straight_through(f):
    def func(x):
        zero = x - stop_gradient(x)
        return  zero + stop_gradient(f(x))
    return func

def straight_through_tuple(f):
    def func(val):
        zero = tree_map(lambda x: x - stop_gradient(x), val)
        return  tree_map(lambda x,y: x + stop_gradient(y), zero, f(val))
    return func

def inject(x,y):
    zero = y - stop_gradient(y)
    return x + zero

def inject_mingrads_pd(x):
    return (x + x.T)/2

def inject_constgrads_pd(x):
    y = x + x.T - diag(diag(x))
    zero = y - stop_gradient(y)
    return stop_gradient(x) + zero

def inv_pd(M):
    M_inv = inv(M)
    return (M_inv + T(M_inv))/2

def Mdot(A, b):
    if A.ndim == b.ndim:
        return jnp.einsum('...ij,...jk->...ik', A, b)
    return jnp.einsum('...ij,...j->...i', A, b)

# def csolve(A, b, sym_pos=False):
#     Ainv = jnp.linalg.inv(A)
#     AinvT = T(Ainv)
#     return jax.lax.custom_linear_solve(lambda x: Mdot(A, x), b, 
#                                      lambda _, x: Mdot(Ainv, x), #jax.scipy.linalg.solve(A, x, sym_pos=sym_pos), 
#                                      lambda _, x: Mdot(AinvT, x),
#                                      symmetric=sym_pos, has_aux=False)

def solve_pd(M, v):
#     return csolve((M + T(M))/2, v)
    return solve((M + T(M))/2, v, assume_a='pos')

def solve_pd_stable(M, v):
    return solve((M + T(M))/2, v, assume_a='pos')

def solve_pds(M, *v):
    return [solve_pd(M, b) for b in v]
#     A = (M + T(M))/2
#     Ainv = jnp.linalg.inv(A)
#     AinvT = T(Ainv)
#     return [jax.lax.custom_linear_solve(lambda x: Mdot(A, x), b, 
#                                      lambda _, x: Mdot(Ainv, x),
#                                      lambda _, x: Mdot(AinvT, x),
#                                      symmetric=False, has_aux=False) for b in v]


def softminus(val):
    return softplus_inverse(val)

def make_prior_fun(prior, logZ_fun, es_fun):
    prior_logZ = logZ_fun(prior)
    if isinstance(prior, tuple):
        def prior_loss(param):
            base = tree_map(lambda x, y, z: jnp_sum((x-y) * z), param, prior, es_fun(param))
            return sum(base) - logZ_fun(param) + prior_logZ
    else:
        def prior_loss(param):
            base = jnp_sum((param - prior) * es_fun(param))
            return base - logZ_fun(param) + prior_logZ
    return prior_loss

def sample_and_logprob(prior, param, logZ_fun, sample_fun, n=1):
    samples = sample_fun(param, n=n)
    if isinstance(prior, tuple):
        get_logprob = lambda samples: sum(tree_map(lambda x, y, z: jnp_sum((x-y) * z), param, prior, samples))
    else:
        get_logprob = lambda samples: jnp_sum((param - prior) * samples)
    base = vmap(get_logprob)(samples)
    return samples, base - logZ_fun(param) + logZ_fun(prior)

def sample_and_logprob_key(prior, param, logZ_fun, sample_fun, key, n=1):
    samples = sample_fun(param, key=key, n=n)
    if isinstance(prior, tuple):
        get_logprob = lambda samples: sum(tree_map(lambda x, y, z: jnp_sum((x-y) * z), param, prior, samples))
    else:
        get_logprob = lambda samples: jnp_sum((param - prior) * samples)
    base = vmap(get_logprob)(samples)
    return samples, base - logZ_fun(param) + logZ_fun(prior)

def mask_potentials(potentials, mask):
    shape = potentials[0].shape[:-2]
    return (where(mask.reshape(shape + (1,1)), potentials[0], zeros_like(potentials[0])),
            where(mask.reshape(shape + (1,1)), potentials[1], zeros_like(potentials[1])))

def cholesky_param(L):
    dim_L = L.shape[-1]
    diag_inds = diag_indices(dim_L)
    L_pos = jnp.tril(L.at[diag_inds].set(softplus(L[diag_inds])))
    return L_pos.dot(L_pos.T) + identity(dim_L) * 1e-8

def cholesky_param_inv(M):
    latent_D = M.shape[-1]
    L_pos = cholesky(M - identity(latent_D) * 1e-8)
    diag_inds = diag_indices(latent_D)
    return L_pos.at[diag_inds].set(softminus(L_pos[diag_inds]))

def diag_param(L):
    latent_D = L.shape[-1]
    diag_inds = diag_indices(latent_D)
    return identity(latent_D) * softplus(L[diag_inds])

def diag_param_inv(M):
    latent_D = M.shape[-1]
    diag_inds = diag_indices(latent_D)
    return identity(latent_D) * softminus(M[diag_inds])

def corr_param(L):
    latent_D = L.shape[0]
    tril_inds = jnp.tril_indices(latent_D, -1)
    diag_inds = jnp.diag_indices(latent_D)
    dev = softplus(L[diag_inds])
    L = L[tril_inds]
    Corr = tfb.Chain([tfb.CholeskyOuterProduct(), tfb.CorrelationCholesky()]).forward(L)
    dev = jnp.expand_dims(dev, -1)
    return dev * Corr * dev.T

def corr_param_inv(M):
    latent_D = M.shape[0]
    tril_inds = jnp.tril_indices(latent_D, -1)
    diag_inds = jnp.diag_indices(latent_D)
    
    dev = jnp.expand_dims(jnp.sqrt(M[diag_inds]), -1)
    
    Corr = M / dev / dev.T
    L = jnp.zeros_like(M)
    L = L.at[tril_inds].set(tfb.Chain([ tfb.CholeskyOuterProduct(), tfb.CorrelationCholesky()]).inverse(Corr))
    udev = softminus(dev.squeeze(-1))
    L = L.at[diag_inds].set(udev)
    return L

def corr_param_rev(L):
    return new_corr_param(L)[::-1, ::-1]

def corr_param_rev_inv(M):
    return new_corr_param_inv(M[::-1, ::-1])

def spectral_param(x):
    vals, vecs = eigh(x)
    vals = jax.nn.softplus(vals)
    return jnp.dot(vals * vecs, vecs.T)

def spectral_param_inv(x):
    vals, vecs = eigh(x)
    vals = softminus(vals)
    return jnp.dot(vals * vecs, vecs.T)

def pd_param(L):
    if 'SVAE_PD_PARAM' in os.environ:
        if os.environ['SVAE_PD_PARAM'] == 'cholesky':
            return cholesky_param(L)
        elif os.environ['SVAE_PD_PARAM'] == 'diag':
            return diag_param(L)
        elif os.environ['SVAE_PD_PARAM'] == 'spectral':
            return spectral_param(L)
        elif os.environ['SVAE_PD_PARAM'] == 'corr':
            return corr_param(L)
        elif os.environ['SVAE_PD_PARAM'] == 'corr_rev':
            return corr_param_rev(L)
    return corr_param(L)

def pd_param_inv(M):
    if 'SVAE_PD_PARAM' in os.environ:
        if os.environ['SVAE_PD_PARAM'] == 'cholesky':
            return cholesky_param_inv(M)
        elif os.environ['SVAE_PD_PARAM'] == 'diag':
            return diag_param_inv(L)
        elif os.environ['SVAE_PD_PARAM'] == 'spectral':
            return spectral_param_inv(M)
        elif os.environ['SVAE_PD_PARAM'] == 'corr':
            return corr_param_inv(M)
        elif os.environ['SVAE_PD_PARAM'] == 'corr_rev':
            return corr_param_rev_inv(L)
    return corr_param_inv(M)

### Flatten and unflatten:

def cat_es_min(param):
    return param[:-1]

def cat_es_to_minflat(param):
    return vmap(cat_es_min)(param).flatten()


def cat_es_from_min(param):
    return pad(param, (0,1), constant_values=1-param.sum())

def cat_es_from_minflat(param, N):
    return vmap(cat_es_from_min)(param.reshape((N, -1)))


def cat_param_min(param):
    return param[:-1] - param[-1]

def cat_param_to_minflat(param):
    return vmap(cat_param_min)(param).flatten()

def cat_param_from_min(param):
    return pad(param, (0,1))

def cat_param_from_minflat(param, N):
    return vmap(cat_param_from_min)(param.reshape((N, -1)))

def gaus_es_min(param):
    EXXT, EX, EXXNT = param
    lowert = ((EXXT + EXXT.T)/2)[tril_indices(EXXT.shape[-1])]
    return concatenate([lowert, EX.squeeze(), EXXNT.flatten()])

def gaus_es_to_minflat(param):
    EXXT, EX, EXXNT = param
    all_but_last = vmap(gaus_es_min)((EXXT[:-1], EX[:-1], EXXNT)).flatten()
    last_lowert = ((EXXT[-1] + EXXT[-1].T)/2)[tril_indices(EXXT.shape[-1])]
    return concatenate([all_but_last, last_lowert, EX[-1].squeeze()])

def gaus_es_from_min(param, D):
    numel_EXXT = int(D * (D+1)/2)
    EXXT_flat = param[:numel_EXXT]
    EXXT_lower = zeros((D,D)).at[tril_indices(D)].set(EXXT_flat)
    EXXT = EXXT_lower + tril(EXXT_lower,-1).T
    EX = expand_dims(param[numel_EXXT:numel_EXXT+D], -1)
    EXXNT = param[numel_EXXT+D:].reshape((D,D))
    return EXXT, EX, EXXNT

def gaus_es_from_minflat(param, D):
    numel_last = int(D * (D+1)/2) + D
    all_but_last, last = param[:-numel_last], param[-numel_last:]
    EXXT_butlast, EX_butlast, EXXNT = vmap(gaus_es_from_min, in_axes=(0,None))(all_but_last.reshape((-1, numel_last + int(D ** 2))), D)
    EXXT_flat = last[:-D]
    EXXT_lower = zeros((D,D)).at[tril_indices(D)].set(EXXT_flat)
    EXXT_last = expand_dims(EXXT_lower + tril(EXXT_lower,-1).T, 0) 
    EX_last = expand_dims(param[-D:], (0,-1))
    return concatenate([EXXT_butlast, EXXT_last]), concatenate([EX_butlast, EX_last]), EXXNT

def gaus_param_min(param, es=False):
    EX, EXXT, EXXNT, EXNXNT, EXN = param
    D = EXXT.shape[-1]
    if es:
        return concatenate([gaus_es_min((EXXT, EX, EXXNT)), gaus_es_min((EXNXNT, EXN, EXXNT))])[:-D**2]
    EXXT = inject(inject(EXXT, tril(EXXT, -1)), triu(EXXT, 1)) # Somewhat ugly fix. 
    EXNXNT = inject(inject(EXNXNT, tril(EXNXNT, -1)), triu(EXNXNT, 1)) # Somewhat ugly fix. 
    return concatenate([gaus_es_min((-1/2 * EXXT, -EX, EXXNT)), gaus_es_min((-1/2 * EXNXNT, EXN, EXXNT))])[:-D**2]

def gaus_param_from_min(param, D, es=False):
    cur_param, next_param = split(concatenate([param, zeros(D**2)]), 2)
    EXXT, EX, EXXNT = gaus_es_from_min(cur_param, D)
    EXNXNT, EXN, _ = gaus_es_from_min(next_param, D)
    if es:
        return (EX, EXXT, EXXNT, EXNXNT, EXN)
    EXXT = inject(inject(EXXT, -1/2 * tril(EXXT, -1)), -1/2 * triu(EXXT, 1)) # Somewhat ugly fix.
    EXNXNT = inject(inject(EXNXNT, -1/2 * tril(EXNXNT, -1)), -1/2 * triu(EXNXNT, 1)) # Somewhat ugly fix.
    return (-EX, -2 * EXXT, EXXNT, -2 * EXNXNT, EXN)

def gaus_param_to_minflat(param):
    h1, J11, J12, J22, h2 = param
    new_h1 = -h1.at[1:].set(h1[1:]-h2[:-1])
    new_h1 = concatenate([new_h1, h2[[-1],:]])
    new_J11 = -1/2 * J11.at[1:].set(J11[1:] + J22[:-1])
    new_J11 = concatenate([new_J11, -1/2 * J22[[-1],:]])
    new_J11 = inject(inject(new_J11, tril(new_J11, -1)), triu(new_J11, 1)) # Somewhat ugly fix. 
    return gaus_es_to_minflat((new_J11, new_h1, J12))

def gaus_param_from_minflat(param, D):
    EXXT, EX, J12 = gaus_es_from_minflat(param, D)
    EXXT = inject(inject(EXXT, -1/2 * tril(EXXT, -1)), -1/2 * triu(EXXT, 1)) # Somewhat ugly fix.
    J11 = -2 * EXXT[:-1]
    J22 = zeros_like(J11).at[-1].set(-2 * EXXT[-1])
    h1 = -EX[:-1]
    h2 = zeros_like(h1).at[-1].set(EX[-1])
    return (h1, J11, J12, J22, h2)

def flatten_params(params):
    gaus_param, cat_param = params
    gaus_flat = gaus_param_to_minflat(gaus_param)
    threshold = gaus_flat.shape[0]
    return concatenate([gaus_flat, cat_param_to_minflat(cat_param)]), threshold

def unflatten_params(params, threshold, D, N):
    gaus_param, cat_param = params[:threshold], params[threshold:]
    return (gaus_param_from_minflat(gaus_param, D), cat_param_from_minflat(cat_param, N))

def flatten_es(params):
    gaus_param, cat_param = params
    gaus_flat = gaus_es_to_minflat(gaus_param)
    return concatenate([gaus_flat, cat_es_to_minflat(cat_param)])

def unflatten_es(params, threshold, D, N):
    gaus_param, cat_param = params[:threshold], params[threshold:]
    return (gaus_es_from_minflat(gaus_param, D), cat_es_from_minflat(cat_param, N))

def flatten_es_grouped(gaus_param, cat_param):
    gaus_flat = vmap(gaus_es_to_minflat)(gaus_param).flatten()
    return concatenate((gaus_flat, cat_es_to_minflat(cat_param)))

def unflatten_es_grouped(params, threshold, D, N, groups):
    gaus_groups = jnp.stack(split(params[:groups*threshold], groups, axis=0))
    gaus_param = vmap(gaus_es_from_minflat, in_axes=[0,None])(gaus_groups, D)
    cat_param = cat_es_from_minflat(params[groups*threshold:], N)
    return gaus_param, cat_param

### Sparse Matrices
@jit
def make_sparse_overlapped_block_matrix(blocks, overlap_blocks):
    if type(overlap_blocks) is int:
        overlap_blocks = zeros((blocks.shape[0] - 1, overlap_blocks, overlap_blocks), dtype=blocks.dtype)
    overlaps = overlap_blocks.shape[1]
    underlaps = blocks.shape[1] - overlaps
    middles = underlaps - overlaps

    block_bottoms = blocks[:-1, overlaps:]
    block_tops = blocks[1:, :overlaps]
    block_nums = arange(block_tops.shape[0])

    @vmap
    def write_blocks(bottom, top, overlap_block, num):
        data = bottom[:middles].flatten()
        indices = tile(arange(bottom.shape[-1]), (middles, 1)).flatten()

        top = top.at[:, :overlaps].set(top[:, :overlaps] - overlap_block)
        overlap_data = pad(bottom[-overlaps:], ((0,0), (0, underlaps))) + pad(top, ((0,0), (underlaps, 0)))
        overlap_indices = tile(arange(overlap_data.shape[-1]), (overlap_data.shape[0], 1))

        indptr = arange(middles) * bottom.shape[-1]
        overlap_indptr = arange(overlaps) * overlap_data.shape[1] + data.shape[0]
        indptr = concatenate([indptr, overlap_indptr])

        data = concatenate([data, overlap_data.flatten()])
        indices = concatenate([indices, overlap_indices.flatten()]) + num * underlaps
        indptr = num * data.shape[0] + indptr
        return data, indices, indptr

    data, indices, indptr = write_blocks(block_bottoms, block_tops, overlap_blocks, block_nums)
    data, indices, indptr = data.flatten(), indices.flatten(), indptr.flatten()

    first_data = blocks[0, :overlaps].flatten()
    data = concatenate([first_data, data])
    indices = concatenate([tile(arange(blocks.shape[-1]), (overlaps, 1)).flatten(), indices])
    indices = concatenate([indices, tile(arange(blocks.shape[-1]), (underlaps, 1)).flatten() + block_bottoms.shape[0] * underlaps])

    first_indptr = arange(overlaps) * blocks.shape[-1]
    last_indptr = arange(underlaps + 1) * blocks.shape[-1] + data.shape[0]
    data = concatenate([data, blocks[-1, overlaps:].flatten()])
    indptr = concatenate([first_indptr, indptr + first_data.shape[0], last_indptr])
    return data, indices, indptr

def make_csr(blocks, overlap_blocks):
    if type(overlap_blocks) is int:
        overlap_blocks = zeros((blocks.shape[0] - 1, overlap_blocks, overlap_blocks), dtype=blocks.dtype)

    overlaps = overlap_blocks.shape[1]
    underlaps = blocks.shape[1] - overlaps
    shape = blocks.shape[0] * underlaps + overlaps
    data, indices, indptr = make_sparse_overlapped_block_matrix(blocks, overlap_blocks)
    return csr_matrix((data, indices, indptr), shape=(shape, shape))

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

    
def vmap_while_loop(cond_fun, body_fun, init_val, batch_size=-1, chunk_factor=16):
    batch_size = jax.tree_leaves(init_val)[0].shape[0] if batch_size <= 0 else batch_size
    #assert batch_size % chunk_factor == 0

    # Size to swap into working batch when finished
    chunk_size = batch_size // chunk_factor

    # Explicitly stop work on jobs that are finished
    def body(val):
        working = cond_fun(val)
        new_val = body_fun(val)
        return jax.tree_map(lambda new, old: jax.vmap(jnp.where)(working, new, old), new_val, val)

    def all_cond_fun(val):
        return jnp.any(cond_fun(val))

    def half_cond_fun(val):
        nfinished = jnp.sum(jnp.logical_not(cond_fun(val)))
        return nfinished < chunk_size

    def inner_loop(val):
        return jax.lax.while_loop(half_cond_fun, body, val)

    def final_loop(val):
        return jax.lax.while_loop(all_cond_fun, body, val)

    final_val = init_val # Final output will be same size and structure as input

    # The set of values to operate on in parallel at any given time
    working_batch = init_val
    working_batch_size = batch_size
    working_inds = jnp.arange(batch_size)

    # Iterate over the full set of jobs in the batch
    for i in range(1, chunk_factor):
        # Iterate all jobs in the working batch until at least a fraction of 1/chunk_factor are complete
        working_batch = inner_loop(working_batch)

        # Find the first 1/chunk_factor fraction of jobs in the working batch that are complete
        finished = jnp.logical_not(cond_fun(working_batch))
        finished = jnp.logical_and(jnp.cumsum(finished, 0) <= chunk_size, finished)

        unfinished = jnp.argwhere(jnp.logical_not(finished), size=working_batch_size - chunk_size)[:, 0]
        finished = jnp.argwhere(finished, size=chunk_size)[:, 0]
        finished_inds = working_inds[finished]

        # Swap finished jobs into the output batch
        final_val = jax.tree_map(lambda x, y: x.at[finished_inds].set(y[finished]), final_val, working_batch)

        # Shrink the working batch
        working_batch = jax.tree_map(lambda x: x[unfinished], working_batch)
        working_batch_size = working_batch_size - chunk_size
        working_inds = working_inds[unfinished]

    # Once the last set of jobs has been added to the working batch, iterate until all outstanding jobs are finished
    working_batch = final_loop(working_batch)

    final_val = jax.tree_map(lambda x, y: x.at[working_inds].set(y), final_val, working_batch)
    return final_val