from jax import jit, vmap, custom_vjp, jvp
from jax.numpy import log, arange, pi, dot, matmul, expand_dims, identity, diag_indices
from jax.numpy.linalg import slogdet, cholesky
from jax.scipy.special import multigammaln, digamma
from utils import inv_pd, softminus, pd_param, pd_param_inv
from flax.linen import softplus
import tensorflow_probability.substrates.jax.distributions as tfd
# Normal-Inverse-Wishart Distribution.
# S is a D x D matrix, loc is a D x 1 vector, lam and nu are scalars.
# Using natural parameters
#    S + lam * loc loc^T
#    lam * loc
#    lam      
#    nu + p + 2
#
# and sufficient statistics
#    - tau / 2
#    tau * mu
#    - mu^T tau mu / 2
#    ln |tau| / 2
#
# NOT automatically batched

@jit
def nat_to_moment(natparam):
    A, b, lam, d = natparam
    loc = b/lam
    return A - loc.dot(b.T), loc, lam,  d - A.shape[-1] - 2

@jit
def moment_to_nat(params):
    S, loc, lam, nu = params
    return S + lam * loc.dot(loc.T), lam * loc, lam, nu + S.shape[-1] + 2

@jit
def expected_stats(natparam):
    S, loc, lam, nu = nat_to_moment(natparam)
    p = S.shape[-1]
    S_inv = inv_pd(S)
    E_tau = nu * S_inv
    E_tau_mu = E_tau.dot(loc)
    E_muT_tau_mu = p / lam + loc.T.dot(E_tau_mu).squeeze()
    E_logdet_tau = digamma((nu - arange(p))/2).sum() + p * log(2) - slogdet(S)[1]
    return -E_tau/2, E_tau_mu, -E_muT_tau_mu/2, E_logdet_tau/2

@jit
def logZ(natparam):
    S, loc, lam, nu = nat_to_moment(natparam)
    p = S.shape[-1]
    const1 = log(2) * p
    const2 = log(2 * pi) * p/2
    return nu/2 * (const1 - slogdet(S)[1]) + multigammaln(nu/2, p) - p/2 * log(lam) + const2

def sample(natparam, key, n=1):
    S, loc, lam, nu = nat_to_moment(natparam)
    precision = tfd.WishartTriL(df=nu, scale_tril=cholesky(inv_pd(S))).sample(n, seed=key)
    return precision, tfd.MultivariateNormalFullCovariance(loc[:,0], inv_pd(precision)).sample(seed=key)

def sample_es(natparam, key, n=1):
    precision, val = sample(natparam, key, n)
    neg_half_tau = -1/2 * precision
    log_det_tau = slogdet(precision)[1]/2
    tau_mu = vmap(matmul)(precision, val)
    muT_tau_mu = -1/2 * vmap(dot)(tau_mu, val)
    return neg_half_tau, expand_dims(tau_mu, -1), muT_tau_mu, log_det_tau

#unconstrained to natural
def uton(params):
    S_p, loc, lam_p, nu_p = params
    latent_D = S_p.shape[-1]
    S = pd_param(S_p) + identity(latent_D) * 1e-6
    lam = softplus(lam_p)
    nu = softplus(nu_p) + latent_D - 1
    return moment_to_nat((S, loc, lam, nu))

# natural to unconstrained
def ntou(natparam):
    S, loc, lam, nu = nat_to_moment(natparam)
    latent_D = S.shape[-1]

    S_p = pd_param_inv(S - identity(latent_D) * 1e-6)
    lam_p = softminus(lam)
    nu_p = softminus(nu - latent_D + 1)
    return S_p, loc, lam_p, nu_p

uton_natgrad = custom_vjp(uton)

def uton_natgrad_fwd(params):
    return uton(params), uton(params)

def uton_natgrad_bwd(resids, grads):
    return (jvp(ntou, (resids,), (grads,))[1],)

uton_natgrad.defvjp(uton_natgrad_fwd, uton_natgrad_bwd)