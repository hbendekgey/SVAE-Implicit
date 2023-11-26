from jax import jit, vmap, jacrev, custom_vjp, vjp, tree_map, value_and_grad
from jax.lax import scan, stop_gradient, while_loop, associative_scan, cond
from jax.numpy import concatenate, expand_dims, log, pi, tile, zeros_like, ones_like, tensordot, arange, repeat, inf, abs, logical_or, logical_and
from jax.numpy.linalg import slogdet, cholesky
from jax.random import normal as rand_norm
from jax.random import split
from jax.scipy.special import logsumexp
from jax.scipy.linalg import solve_triangular
from distributions import normal, categorical, mniw, dirichlet, niw
from utils import solve_pd, solve_pds, inv_pd, cat_param_min, cat_param_from_min, cat_es_min, inject, make_csr, gaus_param_min, gaus_param_from_min, T, flatten_es, unflatten_es, binom
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from jax.experimental.host_callback import call, id_print, id_tap
from time import time
import jax.numpy as jnp
from jax import nn
import jax
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.math import log_cumsum_exp
from numpy import infty, identity

def jit(x, *args, **kwargs):
    return x

### LDS BP Algorithms

@jit
def forward_predict(J, h, h1, J11, J12, J22, h2, logZ = 0):
    P = J + J11
    J12TP_inv = solve_pd(P, J12).T
    J_new = J22 - J12TP_inv.dot(J12)
    h_new = h2 + J12TP_inv.dot(h - h1)
    logZ = logZ + 1/2 * ((h - h1).T.dot(solve_pd(P,h - h1)).squeeze() - slogdet(P)[1])
    return J_new, h_new, logZ

@jit
def forward_measurement(J, h, R, r):
    return J + R, h + r

@jit
def lds_forward_iter(carry, slices):
    Jpi, hpi, logZ = carry
    Ri, ri, h1i, J11i, J12i, J22i, h2i = slices 
    Jfi, hfi = forward_measurement(Jpi, hpi, Ri, ri)
    Jpi, hpi, logZ = forward_predict(Jfi, hfi, h1i, J11i,
                                     J12i, J22i, h2i, logZ)
    return (Jpi, hpi, logZ), (Jfi, hfi, Jpi, hpi)

@jit 
def lds_backward_iter(carry, slices):
    Jns, hns, mun = carry
    Jnp, hnp, Jf, hf, h1, J11, J12, J22, h2 = slices
    C = Jns - Jnp + J22
    J12C_inv = solve_pd(C, J12.T).T
    Js = Jf + J11 - J12C_inv.dot(J12.T)
    hs = hf - h1 + J12C_inv.dot(hns - hnp + h2)
    mu, var = normal.Jh_to_moment((Js, hs))
    return (Js, hs, mu), (var + mu * mu.T, mu, var.dot(J12C_inv) + mu * mun.T, Js, hs)

@jit
def lds_inference_sequential(recog_potentials, init, transition_params):
    neg_halfR, r = recog_potentials
    R = -2 * neg_halfR
    h1, J11, J12, J22, h2 = transition_params

    # filter forward
    N = R.shape[0]
    logZ = 0.

    carry = init[0], init[1], logZ
    slices = R[:-1], r[:-1], h1, J11, J12, J22, h2
    (_, _, logZ), (Jf, hf, Jp, hp) = scan(lds_forward_iter, carry, slices)
    Jp = concatenate([expand_dims(init[0], 0), Jp], 0)
    hp = concatenate([expand_dims(init[1], 0), hp], 0)
    Jfi, hfi = forward_measurement(Jp[-1], hp[-1], R[-1], r[-1])
    Jf = concatenate([Jf, expand_dims(Jfi, 0)], 0)
    hf = concatenate([hf, expand_dims(hfi, 0)], 0)

    J_inv_h = solve_pd(Jf[-1],hf[-1])
    logZ = logZ + 1/2 * (hf[-1].T.dot(J_inv_h).squeeze() - slogdet(Jf[-1])[1]) + log(2 * pi)*R.shape[-1]*N/2

    # smoother backward
    EX, var = normal.Jh_to_moment((Jf[-1], hf[-1]))
    EXXT = var + EX * EX.T

    carry = Jf[-1], hf[-1], EX
    slices = Jp[1:], hp[1:], Jf[:-1], hf[:-1], h1, J11, J12, J22, h2
    _, (EXXTs, EXs, EXXNTs, Js, hs) = scan(lds_backward_iter, carry, slices, reverse=True)
    Js = concatenate([Js, expand_dims(Jf[-1], 0)], 0)
    hs = concatenate([hs, expand_dims(hf[-1], 0)], 0)
    EXXTs = concatenate([EXXTs, expand_dims(EXXT, 0)], 0)
    EXs = concatenate([EXs, expand_dims(EX, 0)], 0)
    
    return (EXXTs, EXs, EXXNTs), logZ, (Jf[:-1], hf[:-1], Js[1:], hs[1:])

def lds_associative_scan_fwd(messages_i, messages_j):
    (h1, J11, J12, J22, h2, Gi, gi) = messages_i
    (f1, F11, F12, F22, f2, Gj, gj) = messages_j

    J12C_inv = solve_pd(Gj + J22, J12.T).T
    gij = -h1 + J12C_inv.dot(gj + h2)
    Gij = J11 - J12C_inv.dot(J12.T)

    #J12P_inv = solve_pd(Gj + J22 + F11, J12.T).T
    #F12TP_inv = solve_pd(Gj + J22 + F11, F12).T
    
    J12P_inv, F12TP_inv = solve_pds(Gj + J22 + F11, J12.T, F12)
    J12P_inv, F12TP_inv = J12P_inv.T, F12TP_inv.T

    Fij_1 = h1 - J12P_inv.dot(h2 - f1 + gj) + gij
    Fij_11 = J11 - J12P_inv.dot(J12.T) - Gij
    Fij_12 = J12P_inv.dot(F12)
    Fij_2 = f2 + F12TP_inv.dot(h2 - f1 + gj)
    Fij_22 = F22 - F12TP_inv.dot(F12)

    return (Fij_1, Fij_11, Fij_12, Fij_22, Fij_2, Gi + Gij, gi + gij)

def lds_associative_scan_bwd(messages_i, messages_j):
    (h1, J11, J12, J22, h2) = messages_j
    (f1, F11, F12, F22, f2) = messages_i

    #J12P_inv = solve_pd(J22 + F11, J12.T).T
    #F12TP_inv = solve_pd(J22 + F11, F12).T
    J12P_inv, F12TP_inv = solve_pds(J22 + F11, J12.T, F12)
    J12P_inv, F12TP_inv = J12P_inv.T, F12TP_inv.T

    Fij_1 = h1 - J12P_inv.dot(h2 - f1)
    Fij_11 = J11 - J12P_inv.dot(J12.T)
    Fij_12 = J12P_inv.dot(F12)
    Fij_2 = f2 + F12TP_inv.dot(h2 - f1)
    Fij_22 = F22 - F12TP_inv.dot(F12)

    return (Fij_1, Fij_11, Fij_12, Fij_22, Fij_2)

@jit
def lds_inference(recog_potentials, init, transition_params):
    # unpack params
    neg_halfR, r = recog_potentials
    R = -2 * neg_halfR
    init_J, init_h = init
    h1, J11, J12, J22, h2 = transition_params

    # get forward factors in parallel
    def get_g(h1, J11, J12, J22, h2):
        J12C_inv = solve_pd(J22, J12.T).T
        return (J11 - J12C_inv.dot(J12.T), -h1 + J12C_inv.dot(h2))
    g = vmap(get_g)(h1, J11, J12, J22 + R[1:], h2 + r[1:])
    g = tree_map(lambda x: concatenate([zeros_like(x[:1]), x], 0), g)
    h1_full, J11_full, J12_full, J22_full, h2_full = tree_map(lambda x: concatenate([zeros_like(x[:1]), x], 0), transition_params)
    J22_full, h2_full = J22_full.at[0].set(init_J), h2_full.at[0].set(init_h)
    f = (h1_full + g[1], J11_full - g[0], J12_full, J22_full + R, h2_full + r)

    # forward message passing
    Jf, hf = associative_scan(vmap(lds_associative_scan_fwd), f + g)[3:5]

    # compute log partition function
    def get_logZ(h, P):
        return 1/2 * (h.T.dot(solve_pd(P,h)).squeeze() - slogdet(P)[1])
    logZ = vmap(get_logZ)(hf[:-1] - h1, Jf[:-1] + J11).sum() + get_logZ(hf[-1], Jf[-1]) + log(2 * pi)*R.shape[-1]*R.shape[0]/2

    # get backward factors in parallel
    a = (h1 - hf[:-1], J11 + Jf[:-1], J12, J22 + R[1:] - Jf[1:], h2 + r[1:] - hf[1:])
    final_message = (-hf[-1:], Jf[-1:], zeros_like(J12[-1:]), zeros_like(J22[-1:]), zeros_like(h2[-1:]))
    a = tree_map(lambda x,y: concatenate([x,y], 0), a, final_message)
    hs, Js = associative_scan(vmap(lds_associative_scan_bwd), a, reverse=True)[:2]

    # get expected statistics
    mu, var = vmap(normal.Jh_to_moment)((Js, -hs))
    C = Js[1:] - (Jf[1:]-R[1:]) + J22
    def get_es(mu, var, mun, C, J12):
        J12C_inv = solve_pd(C, J12.T).T
        return (var.dot(J12C_inv) + mu * mun.T)
    EXXNT = vmap(get_es)(mu[:-1], var[:-1], mu[1:], C, J12)
    EXXT = vmap(lambda mu, var: var + mu * mu.T)(mu, var)

    return (EXXT, mu, EXXNT), logZ, (Jf[:-1], hf[:-1], Js[1:], -hs[1:])

def sample_mvn_information(J, h, epsilon):
    mu, lam = solve_pd(J,h), J
    L_inv = cholesky(lam).T
    return mu + solve_triangular(L_inv, epsilon)

@jit
def lds_backward_iter_and_sample(carry, slices):
    Jns, hns, mun, zn = carry
    Jnp, hnp, Jf, hf, h1, J11, J12, J22, h2, epsilon = slices
    C = Jns - Jnp + J22
    J12C_inv = solve_pd(C, J12.T).T
    Js = Jf + J11 - J12C_inv.dot(J12.T)
    hs = hf - h1 + J12C_inv.dot(hns - hnp + h2)
    mu, var = normal.Jh_to_moment((Js, hs))

    sample_J = Jf + J11
    sample_h = hf + J12.dot(zn) - h1
    z = sample_mvn_information(sample_J, sample_h, epsilon)

    return (Js, hs, mu, z), (var + mu * mu.T, mu, var.dot(J12C_inv) + mu * mun.T, Js, hs, z)

@jit
def lds_inference_and_sample(recog_potentials, init, transition_params, key):
    # unpack params
    neg_halfR, r = recog_potentials
    R = -2 * neg_halfR
    init_J, init_h = init
    h1, J11, J12, J22, h2 = transition_params

    # get forward factors in parallel
    def get_g(h1, J11, J12, J22, h2):
        J12C_inv = solve_pd(J22, J12.T).T
        return (J11 - J12C_inv.dot(J12.T), -h1 + J12C_inv.dot(h2))
    g = vmap(get_g)(h1, J11, J12, J22 + R[1:], h2 + r[1:])
    g = tree_map(lambda x: concatenate([zeros_like(x[:1]), x], 0), g)
    h1_full, J11_full, J12_full, J22_full, h2_full = tree_map(lambda x: concatenate([zeros_like(x[:1]), x], 0), transition_params)
    J22_full, h2_full = J22_full.at[0].set(init_J), h2_full.at[0].set(init_h)
    f = (h1_full + g[1], J11_full - g[0], J12_full, J22_full + R, h2_full + r)

    # forward message passing
    Jf, hf = associative_scan(vmap(lds_associative_scan_fwd), f + g)[3:5]

    # compute log partition function
    def get_logZ(h, P):
        return 1/2 * (h.T.dot(solve_pd(P,h)).squeeze() - slogdet(P)[1])
    logZ = vmap(get_logZ)(hf[:-1] - h1, Jf[:-1] + J11).sum() + get_logZ(hf[-1], Jf[-1]) + log(2 * pi)*R.shape[-1]*R.shape[0]/2

    # smoother backward
    EX, var = normal.Jh_to_moment((Jf[-1], hf[-1]))
    epsilon = rand_norm(key, hf.shape).astype(R.dtype)

    EXXT = var + EX * EX.T
    z = sample_mvn_information(Jf[-1], hf[-1], epsilon[-1])
    Jp, hp = vmap(forward_measurement)(Jf, hf, -R, -r)
    carry = Jf[-1], hf[-1], EX, z
    slices = Jp[1:], hp[1:], Jf[:-1], hf[:-1], h1, J11, J12, J22, h2, epsilon[:-1]
    _, (EXXTs, EXs, EXXNTs, Js, hs, zs) = scan(lds_backward_iter_and_sample, carry, slices, reverse=True)
    Js = concatenate([Js, expand_dims(Jf[-1], 0)], 0)
    hs = concatenate([hs, expand_dims(hf[-1], 0)], 0)
    EXXTs = concatenate([EXXTs, expand_dims(EXXT, 0)], 0)
    EXs = concatenate([EXs, expand_dims(EX, 0)], 0)
    zs = concatenate([zs, expand_dims(z, 0)], 0)

    return (EXXTs, EXs, EXXNTs), logZ, zs

@jit
def lds_inference_homog(recog_potentials, init, transition_params):
    N = recog_potentials[0].shape[0]
    def rep(param):
        return tile(param, (N-1,1,1))
    return lds_inference(recog_potentials, init, tree_map(rep, transition_params))

def sample_mvn(mu, var, epsilon):
    return (mu + cholesky(var).dot(epsilon))

def sample_backward_iter(carry, slices):
    xn, mun, varn = carry
    mu, var, covar, epsilon = slices
    J12C_inv = solve_pd(varn, covar.T).T
    sample_mu = mu + J12C_inv.dot(xn - mun)
    sample_var = var - J12C_inv.dot(covar.T)
    sample = sample_mvn(sample_mu, sample_var, epsilon)
    return (sample, mu, var), sample

@jit
def sample_lds(gaus_expected_stats, key):
    EXXT, EX, EXXNT = gaus_expected_stats
    N = EX.shape[0]
    mu, var, covar = EX, EXXT - EX * T(EX), EXXNT - EX[:-1] * T(EX[1:])
    epsilon = rand_norm(key, mu.shape)
    xN = sample_mvn(mu[-1], var[-1], epsilon[-1])
    init = xN, mu[-1], var[-1]
    slices = mu[:-1], var[:-1], covar, epsilon[:-1]
    _, samples = scan(sample_backward_iter, init, slices, reverse=True)
    samples = concatenate([samples, expand_dims(xN, 0)], 0)
    return samples.squeeze(-1)

@jit
def lds_expected_stats_from_potentials(potentials):
    _, EX = normal.expected_stats_masked(potentials)
    return EX * T(EX), EX, EX[:-1] * T(EX)[1:]

@jit
def lds_transition_params_to_nat(init, trans):
    init_nat = -1/2 * init[0], init[1]
    trans_nat = trans[0] * -1, trans[1] * -1/2, trans[2], trans[3] * -1/2, trans[4]
    return init_nat, trans_nat

### HMM BP Algorithm

@jit
def hmm_forward_iter(carry, slices):
    fmessage = carry
    obs_lp, trans_lps = slices
    new_message = obs_lp + logsumexp(fmessage + trans_lps, 0, keepdims=True).T
    return new_message, new_message

@jit
def hmm_backward_iter(carry, slices):
    bmessage = carry
    obs_lp, trans_lps = slices
    new_message = logsumexp(obs_lp + trans_lps + bmessage, -1, keepdims=True).T
    return new_message, new_message

# init_lps is K x 1, trans_lps is K x K, obs_lps is T x K
@jit
def hmm_inference(init_lps, trans_lps, obs_lps):
    obs_lps = expand_dims(obs_lps, -1)
    N = obs_lps.shape[0]
    tiled_params = tile(expand_dims(trans_lps, 0), (N-1, 1, 1))
    carry = obs_lps[0] + init_lps
    slices = obs_lps[1:], tiled_params
    _, fmessages = scan(hmm_forward_iter, carry, slices)
    logZ = logsumexp(fmessages[-1],0).squeeze()
    fmessages = concatenate([expand_dims(carry, 0), fmessages], 0).squeeze(-1)

    obs_lps_T = obs_lps.swapaxes(-1,-2)[1:]
    carry = zeros_like(obs_lps_T[-1])
    slices = obs_lps_T, tiled_params
    _, bmessages = scan(hmm_backward_iter, carry, slices, reverse=True)
    bmessages = concatenate([bmessages, expand_dims(carry, 0)], 0).squeeze(-2)
    cat_es = categorical.expected_stats(fmessages + bmessages)
    return cat_es, logZ, (fmessages, bmessages)

# Hidden Semi-Markov model inference
def hsmm_forward_iter(carry, slices):
    fmessage = carry # M x K x 1
    obs_lp, trans_lps, not_self_trans_lps = slices # K x 1, K x K, M x K x 1

    jump_message = logsumexp(fmessage + not_self_trans_lps, 0) # K x 1
    jump = hmm_forward_iter(jump_message, (obs_lp, trans_lps))[0] # K x 1
    stay = fmessage[:-1] + expand_dims(obs_lp, 0) # M-1 x K x 1

    new_message = fmessage.at[0].set(jump).at[1:].set(stay)
    return new_message, new_message

def hsmm_backward_iter(carry, slices):
    bmessage = carry # M x K x 1
    obs_lp, trans_lps, not_self_trans_lps = slices # K x 1, K x K, M x K x 1
    
    jump_message = bmessage[0]
    jump = hmm_backward_iter(jump_message.T, (obs_lp.T, trans_lps))[0].T # K x 1
    jump = expand_dims(jump, 0) + not_self_trans_lps # M x K x 1

    stay = bmessage[1:] + expand_dims(obs_lp, 0) # M-1 x K x 1
    
    new_message = jnp.logaddexp(zeros_like(bmessage).at[:-1].set(stay), jump)
    return new_message, new_message

# init_lps is K x 1, trans_lps is K x K-1, self_trans_lps is M x K, obs_lps is T x K
def hsmm_inference(init_lps, trans_lps, not_self_trans_lps, obs_lps):
    T = obs_lps.shape[0]
    M = not_self_trans_lps.shape[0]
    K = trans_lps.shape[0]

    # reshape messages for forward pass to be T x K x 1
    obs_lps = jnp.expand_dims(obs_lps, -1)

    # tile transition parameters, adding -infinity self transition to the matrix. (T-1) x K x K
    trans_mask = (1-identity(K)).astype(bool)
    tiled_trans = (-jnp.ones((T-1,K,K), trans_lps.dtype) * infty).at[:,trans_mask].set(trans_lps.flatten())

    # self transition probabilities, (T-1) x M x K x 1
    tiled_self_trans = expand_dims(tile(not_self_trans_lps, (T-1,1,1)),-1)
    ending_lps = expand_dims(jnp.flip(log_cumsum_exp(jnp.flip(not_self_trans_lps,0), 0),0),-1)

    # make M x K x 1 initialized marginals
    carry = tile(obs_lps[0] + init_lps, (M,1,1)).at[1:].set(-infty)
    slices = obs_lps[1:], tiled_trans, tiled_self_trans

    # forward
    _, fmessages = scan(hsmm_forward_iter, carry, slices) # T-1 x M x K x 1
    logZ = logsumexp(fmessages[-1] + ending_lps)
    fmessages = concatenate([expand_dims(carry, 0), fmessages], 0).squeeze(-1)

    # backward parameters
    carry = ending_lps # M x K x 1 
    _, bmessages = scan(hsmm_backward_iter, carry, slices, reverse=True) # (T-1) x M x K x 1
    bmessages = concatenate([bmessages, expand_dims(carry, 0)], 0).squeeze(-1) # T x M x K
    cat_es = categorical.expected_stats(logsumexp(fmessages + bmessages, 1))
    return cat_es, logZ, (fmessages, bmessages)

def hmm_sample_bwd(carry, slices):
    state = carry
    trans, key, fmessage = slices
    new_state = tfd.Categorical(logits=trans[:,state] + fmessage).sample(seed=key)
    return new_state, new_state

def hmm_sample(cat_es, trans, key, fmessages):
    end_key, bwd_key = split(key)
    end_state = tfd.Categorical(probs=cat_es[-1]).sample(seed=end_key)
    fmessages = fmessages[:-1]
    slices = tile(trans, (fmessages.shape[0],1,1)), split(bwd_key, fmessages.shape[0]), fmessages
    carry = end_state
    _, states = scan(hmm_sample_bwd, carry, slices, reverse=True)
    embed_state = lambda x: jnp.zeros_like(trans[0]).at[x].set(1)
    return jnp.concatenate([vmap(embed_state)(states), jnp.expand_dims(embed_state(end_state), -2)], -2)


def hmm_sample_fwd(carry, slices):
    state = carry
    trans, key = slices
    new_state = tfd.Categorical(logits=trans[state]).sample(seed=key)
    return new_state, new_state

def hmm_forecast(final_probs, trans, key, forecast_length):
    init_key, iter_key = split(key)
    slices = tile(trans, (forecast_length,1,1)), split(iter_key, forecast_length)
    carry = tfd.Categorical(probs=final_probs).sample(seed=init_key)
    _, states = scan(hmm_sample_fwd, carry, slices)
    embed_state = lambda x: jnp.zeros_like(final_probs).at[x].set(1)
    return vmap(embed_state)(states)

def lds_forecast_iter(z, slices):
    A, b, var, epsilon = slices
    z_next = jnp.matmul(A,z) + b + jnp.matmul(var, epsilon)
    return z_next, z_next

def slds_forecast(final_Z, cat_expected_stats, mniw_params,
                  trans_params, n_forecast, forecast_rng, cat_forecast=None):
    if cat_forecast is None:
        hmm_rng, lds_rng = split(forecast_rng)
        global_rng, sample_rng = split(hmm_rng, 2)
        E_trans_lps = dirichlet.sample_es(trans_params, global_rng)[0]
        hmm_extend = hmm_forecast(cat_expected_stats[-1], E_trans_lps, sample_rng, n_forecast)
    else:
        lds_rng = forecast_rng
        hmm_extend = cat_forecast

    global_rng, sample_rng = split(lds_rng, 2)
    global_rng = split(global_rng, mniw_params[0].shape[0])
    precisions, Xs = vmap(mniw.sample)(mniw_params, global_rng)
    precisions = tensordot(hmm_extend, precisions.squeeze(1), axes=([1],[0]))
    full_var = vmap(inv_pd)(precisions)
    Xs = tensordot(hmm_extend, Xs.squeeze(1), axes=([1],[0]))
    full_A, full_b = Xs[:,:,:-1], Xs[:,:,-1]
    epsilon = rand_norm(sample_rng, (n_forecast,) + final_Z.shape)
    slices = (full_A, full_b, full_var, epsilon)
    _, new_zs = scan(lds_forecast_iter, final_Z, slices)
    return new_zs

def lds_sample(niw_nat, mniw_params, n_sample, lds_rng, hmm_extend):
    global_rng, init_rng1, init_rng2, sample_rng = split(lds_rng, 4)
    prec, m = niw.sample(niw_nat, init_rng1)
    init_Z = normal.sample((-1/2 * prec[0], m.swapaxes(0,1)), init_rng2)[0]
    global_rng = split(global_rng, mniw_params[0].shape[0])
    precisions, Xs = vmap(mniw.sample)(mniw_params, global_rng)
    precisions = tensordot(hmm_extend, precisions.squeeze(1), axes=([1],[0]))
    full_var = vmap(inv_pd)(precisions)
    Xs = tensordot(hmm_extend, Xs.squeeze(1), axes=([1],[0]))
    full_A, full_b = Xs[:,:,:-1], Xs[:,:,-1]
    epsilon = rand_norm(sample_rng, (n_sample,) + init_Z.shape)
    slices = (full_A, full_b, full_var, epsilon)
    _, new_zs = scan(lds_forecast_iter, init_Z, slices)
    return new_zs

def slds_sample(global_natparams, n_sample, forecast_rng, cat_forecast=None):
    niw_nat, mniw_nat, kappa_nat, alpha_nat = global_natparams
    if cat_forecast is None:
        hmm_rng, lds_rng = split(forecast_rng)
        global_rng, init_rng, sample_rng = split(hmm_rng, 3)
        init_es = dirichlet.sample(kappa_nat.squeeze(), init_rng)[0]
        E_trans_lps = dirichlet.sample_es(alpha_nat, global_rng)[0]
        hmm_extend = hmm_forecast(init_es, E_trans_lps, sample_rng, n_sample)
    else:
        lds_rng = forecast_rng
        hmm_extend = cat_forecast

    return lds_sample(niw_nat, mniw_nat, n_sample, lds_rng, hmm_extend)

def lds_forecast(final_Z, mniw_params, forecast_length, rng):
    key, subkey = split(rng)
    precision, X = jax.tree_map(lambda x: x[0], mniw.sample(mniw_params, key))
    var = inv_pd(precision)
    A,b = X[:,:-1], X[:,-1]
    epsilon = rand_norm(key, (forecast_length,) + final_Z.shape)
    full_A, full_var = jax.tree_map(lambda x: jnp.tile(x, (forecast_length, 1, 1)), (A,var))
    full_b = jnp.tile(b, (forecast_length, 1))
    slices = (full_A, full_b, full_var, epsilon)
    _, new_zs = scan(lds_forecast_iter, final_Z, slices)
    return new_zs

### Trans-HMM BP algorithm.
# A hidden Markov model with emission probabilities/observations registered
# on pairs of discrete states (i.e. transition matrices) instead of individual timesteps.

@jit
def trans_hmm_inference(init_lps, trans_lps):
    slices = jnp.zeros_like(trans_lps[:,0,0]), trans_lps

    carry = init_lps
    _, fmessages = scan(hmm_forward_iter, carry, slices)
    fmessages = concatenate([expand_dims(carry, 0), fmessages], 0).squeeze(-1)
    logZ = logsumexp(fmessages[-1],0).squeeze()

    carry = zeros_like(init_lps.T)
    _, bmessages = scan(hmm_backward_iter, carry, slices, reverse=True)
    bmessages = concatenate([bmessages, expand_dims(carry, 0)], 0).squeeze(-2)
    init_es = jnp.expand_dims(categorical.expected_stats(fmessages[0] + bmessages[0]), -1)
    pairwise_lps = trans_lps + jnp.expand_dims(fmessages[:-1], -1) + jnp.expand_dims(bmessages[1:], -2)
    pairwise_es = nn.softmax(pairwise_lps, axis=[-2, -1])

    return (init_es, pairwise_es), logZ

def trans_hmm_sample(init_es, pairwise_es, key):
    keys = split(key, pairwise_es.shape[0]+1)
    start_state = tfd.Categorical(probs=init_es.squeeze(-1)).sample(seed=keys[0])
    init_sample = jnp.zeros_like(init_es).at[start_state].set(1)

    slices = log(pairwise_es), keys[1:]
    _, states = scan(hmm_sample_fwd, start_state, slices)
    all_states = jnp.concatenate([expand_dims(start_state, 0), states])
    embed_state = lambda x,y: jnp.zeros_like(pairwise_es[0]).at[x,y].set(1)
    embed_marginal = lambda x: jnp.zeros_like(init_es).at[x].set(1)

    return (init_sample, vmap(embed_state)(all_states[:-1], all_states[1:])), vmap(embed_marginal)(all_states)

### Meanfield updates and KL Calculation for SLDS

@jit
def lds_to_hmm_mf(gaus_expected_stats, E_mniw_params):
    EXXT, EX, EXXNT = gaus_expected_stats
    N = EXXT.shape[0]
    es = EX[:-1], EXXT[:-1], EXXNT, EXXT[1:],  EX[1:],   ones_like(EX[:-1,[-1]])
    return sum(tree_map(lambda x, y : tensordot(x,y, axes=([1,2],[1,2])), es, E_mniw_params))

def lds_to_hmm_mf_1step(gaus_es_flat, E_mniw_params):
    D = E_mniw_params[0].shape[1]
    return cat_param_min(sum(map(lambda x, y : tensordot(x,y, axes=([0,1],[1,2])), gaus_param_from_min(gaus_es_flat, D, es=True), E_mniw_params[:-1])) + E_mniw_params[-1].squeeze())

@jit
def hmm_to_lds_mf(cat_es, E_mniw_params, E_init_normalizer, no_const=False):
    expected_param = lambda i: tensordot(cat_es,E_mniw_params[i],axes=([1],[0]))
    h1 = -expected_param(0)
    J11 = -2 * expected_param(1)
    J12 = expected_param(2)
    J22 = -2 * expected_param(3)
    h2 = expected_param(4)
    const = 0 if no_const else log(2 * pi)*J11.shape[-1]/2
    E_trans_normalizer = const -(expected_param(5))
    E_prior_logZ = E_init_normalizer + E_trans_normalizer.sum()
    return (h1, J11, J12, J22, h2), E_prior_logZ

@jit
def lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, logZ):
    neg_halfR, r = recog_potentials
    EXXT, EX, _ = gaus_expected_stats
    return (vmap(jnp.diag)(neg_halfR) * vmap(jnp.diag)(EXXT)).sum() + (r * EX).sum() - logZ + E_prior_logZ

@jit
def lds_kl_surr(recog_potentials, gaus_expected_stats, E_prior_logZ, logZ):
    neg_halfR, r = recog_potentials
    EXXT, EX, _ = gaus_expected_stats
    return -logZ + E_prior_logZ

@jit
def lds_kl_gen(gaus_expected_stats, prior_init, prior_params, E_prior_logZ):
    EXXT, EX, EXXNT = gaus_expected_stats
    params = tree_map(lambda y: -y,  prior_params)
    init_kl = ((-prior_init[0]) * EXXT[0]).sum() + ((-prior_init[1]) * EX[0]).sum()
    es = (EX[:-1], EXXT[:-1], EXXNT, EXXT[1:], EX[1:])
    base_kl = sum(tree_map(lambda x, y: (x * y).sum(), params, es))
    return init_kl + base_kl + E_prior_logZ

@jit
def lds_kl_full(recog_potentials, gaus_expected_stats, prior_init, prior_params,
                inference_init, inference_params, E_prior_logZ, logZ):
    neg_halfR, r = recog_potentials
    EXXT, EX, EXXNT = gaus_expected_stats
    params = tree_map(lambda x,y: x-y, inference_params, prior_params)
    init_kl = ((inference_init[0] - prior_init[0]) * EXXT[0]).sum() + ((inference_init[1] - prior_init[1]) * EX[0]).sum()
    es = (EX[:-1], EXXT[:-1], EXXNT, EXXT[1:], EX[1:])
    base_kl = sum(tree_map(lambda x, y: (x * y).sum(), params, es))
    return init_kl + base_kl + lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, logZ)

@jit
def hmm_kl(cat_natparam, cat_expected_stats, logZ):
    return (cat_natparam * cat_expected_stats).sum() - logZ

@jit
def hmm_kl_gen(cat_expected_stats, E_init_lps_prior, E_trans_lps_prior, EZZNT):
    base_kl_init = ((-E_init_lps_prior).squeeze() * cat_expected_stats[0]).sum()
    base_kl_trans = ((-E_trans_lps_prior) * EZZNT).sum()
    return base_kl_init + base_kl_trans

@jit
def hmm_kl_full(cat_natparam, cat_expected_stats, logZ, E_init_lps_prior, E_trans_lps_prior,
                E_init_lps_inf, E_trans_lps_inf, EZZNT):
    base_kl_init = ((E_init_lps_inf - E_init_lps_prior).squeeze() * cat_expected_stats[0].squeeze()).sum()
    base_kl_trans = ((E_trans_lps_inf - E_trans_lps_prior) * EZZNT).sum()
    return base_kl_init + base_kl_trans + hmm_kl(cat_natparam, cat_expected_stats, logZ)

def trans_hmm_kl(cat_natparam, cat_expected_stats, logZ):
    return sum(tree_map(lambda x,y: (x * y).sum(), cat_natparam, cat_expected_stats)) - logZ

def trans_hmm_kl_full(cat_natparam, cat_expected_stats, logZ, prior_params, inference_params):
    adjusted_params = tree_map(lambda x,y,z: x + y - z, cat_natparam, inference_params, prior_params)
    return trans_hmm_kl(adjusted_params, cat_expected_stats, logZ)

def gaus_to_cat_mf(expected_stats, params, normalizer):
    return sum(tree_map(lambda x, y : tensordot(x,y, axes=([1,2],[1,2])), expected_stats, params)) - normalizer

def single_gaus_kl(expected_stats, params, normalizer, recog_potentials):
    return sum(tree_map(lambda x, y: (x * y).sum(), recog_potentials, expected_stats)) - vmap(normal.logZ)(params).sum() + normalizer.sum()

def single_gaus_kl_det(expected_stats, inference_params, prior_params, normalizer, recog_potentials):
    return sum(tree_map(lambda x,y,z: (x * (y - z)).sum(), expected_stats, inference_params, prior_params)) + single_gaus_kl(expected_stats, inference_params, normalizer, recog_potentials)

def single_gaus_kl_sur(expected_stats, params, normalizer, recog_potentials):
    return sum(tree_map(lambda x,y: (x * y).sum(), expected_stats, params)) + single_gaus_kl(expected_stats, params, normalizer, recog_potentials)

def cat_to_gaus_mf(expected_stats, params, normalizer, recog_potentials):
    params = tree_map(lambda x, y: tensordot(expected_stats, x, axes=([1],[0]))+y, params, recog_potentials)
    return params, jnp.matmul(expected_stats, normalizer)

def single_cat_kl(expected_stats, params, prior_params):
    return (expected_stats * params).sum() - categorical.logZ(params + prior_params).sum()

def single_cat_kl_det(expected_stats, cat_natparam, inf_params, prior_params):
    return (expected_stats * (inf_params - prior_params)).sum() + single_cat_kl(expected_stats, cat_natparam, inf_params)

def jumping_lds_to_hmm_mf(expected_stats, init, E_init_normalizer, jump_params, trans_params):
    a = lds_to_hmm_mf(expected_stats, jump_params)
    d = lds_to_hmm_mf(expected_stats, trans_params)
    init_stats = tree_map(lambda x: expand_dims(x[0],0), expected_stats[:2])
    i = gaus_to_cat_mf(init_stats, init, E_init_normalizer)

    def construct_trans_matrix(a,d):
        return jnp.expand_dims(a,-2) + jnp.diag(d) - jnp.diag(a)
    return jnp.expand_dims(i[0],-1), vmap(construct_trans_matrix)(a,d)

def jumping_hmm_to_lds_mf(expected_stats, init, E_init_normalizer, jump_params, trans_params):
    init_es, pairwise_es = expected_stats
    trans_es = vmap(jnp.diag)(pairwise_es)
    jump_es = pairwise_es.sum(-2) - trans_es

    local_params_t, E_prior_logZ_t = hmm_to_lds_mf(trans_es, trans_params, 0, no_const=True)
    local_params_j, E_prior_logZ_j = hmm_to_lds_mf(jump_es, jump_params, 0)

    local_params = tree_map(lambda x,y: x+y, local_params_t, local_params_j)

    empty_rp = tree_map(lambda x: 0, init)
    (J, h), init_normalizer = cat_to_gaus_mf(init_es.T, init, E_init_normalizer, empty_rp)
    init_params = (-2 * J[0], h[0])

    E_prior_logZ = E_prior_logZ_t + E_prior_logZ_j + init_normalizer.squeeze()
    return (init_params, local_params), E_prior_logZ

# E_self_trans_lps will be (K x) 2, E_dur_n is (K x) maxr
def get_duration_lps(E_self_trans_lps, E_dur_n, max_T):
    ns = jnp.expand_dims(jnp.maximum(1,5 * jnp.arange(E_dur_n.shape[0])),-1) # maxr x -1
    durs = jnp.arange(max_T) # T
    base_measure = jnp.sum(jnp.log(binom(ns + durs - 1, durs)) * jnp.expand_dims(E_dur_n, -1), 0) # max_T
    dur_lps = E_self_trans_lps[0] * durs + (E_dur_n * ns.squeeze()).sum() * E_self_trans_lps[1] + base_measure
    return dur_lps

def sample_cat_matrix(logits, key):
    flat_logits = jnp.reshape(logits, -1)
    sample = tfd.Categorical(logits=flat_logits).sample(seed=key)
    one_hot = jnp.zeros_like(flat_logits).at[sample].set(1)
    return jnp.reshape(one_hot, logits.shape)

def hsmm_stay_sample_bwd(carry, _):
    return jnp.zeros_like(carry).at[:-1].set(carry[1:])

def hsmm_jump_sample_bwd(carry, slices):
    state = carry
    trans, durs, key, fmessage = slices
    next_state = jnp.argmax(state[0])
    logits = fmessage + durs + trans[:, next_state]
    return sample_cat_matrix(logits, key)

def hsmm_sample_bwd(carry, slices):
    new_state = cond(carry[0].sum() > 0, hsmm_jump_sample_bwd, hsmm_stay_sample_bwd, carry, slices)
    return new_state, new_state

def hsmm_sample(messages, trans, durs, key):
    fmessages, bmessages = messages
    
    # Sample ending value of k
    end_key, bwd_key = jax.random.split(key)
    end_state = sample_cat_matrix(fmessages[-1]+bmessages[-1], end_key)
    fmessages = fmessages[:-1]
    T = fmessages.shape[0]
    K = trans.shape[0]

    trans_mask = (1-identity(K)).astype(bool)
    tiled_trans = (-jnp.ones((T,K,K), trans.dtype) * infty).at[:,trans_mask].set(trans.flatten())

    slices = tiled_trans, jnp.tile(durs, (T,1,1)), jax.random.split(bwd_key, fmessages.shape[0]), fmessages
    carry = end_state
    _, states = scan(hsmm_sample_bwd, carry, slices, reverse=True)
    return jnp.concatenate([states, jnp.expand_dims(end_state, -3)], -3)