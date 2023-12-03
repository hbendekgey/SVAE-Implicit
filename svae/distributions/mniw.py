from jax import vmap, custom_vjp, jvp, tree_map
from jax.numpy import log, arange, pi, kron, dot, matmul, identity, diag_indices, zeros, diag
from jax.numpy.linalg import slogdet, cholesky
from jax.scipy.special import multigammaln, digamma
from jax.lax import stop_gradient
from utils import inv_pd, solve_pd_stable, softminus, pd_param, pd_param_inv
from flax.linen import softplus
import jax
import tensorflow_probability.substrates.jax.distributions as tfd

# Matrix-Normal-Inverse-Wishart Distribution.

# Using natural parameters
#    S + M V M^T
#    V * M^T
#    V
#    nu + p + p2 + 1

# and sufficient statistics
#    - Lambda / 2
#    X^T * Lambda
#    - X^T Lambda X / 2
#    ln |Lambda| / 2

# The transition function for our LDS is of the form x_{t+1} ~ N(A_t * x_t + b_t, Sigma_t)
# unlike Matt Johnson et al, who drop the b_t. As a result, we need a prior on all three
# parameters: A,b,Sigma. We place a MNIW prior on X, Sigma where X = [A|b]. Thus the dimensions are:
# X: D x (D+1)        output
# Lambda:  D x D      output precision
# M: D x (D+1)        mean of X
# S: D x D            scale matrix for Lambda
# V: (D+1) x (D+1)    determines precisions of X relative to Lambda (My V is V^-1 in most matrix normal definitions)
#
# Expected sufficient statistics are also decomposable:
# E[Sigma^-1 X] = [ E[Sigma^-1 A] | E[Sigma^-1 b] ]
# E[X^T Sigma^-1 X] = 
#                  [ E[A^T Sigma^-1 A] | E[A^T Sigma^-1 b]
#                  [ E[b^T Sigma^-1 A] | E[b^T Sigma^-1 b]
#

def nat_to_moment(natparam):
    A, b, V, d = natparam
    M = solve_pd_stable(V, b).T
#     M = diag(1/diag(V)).dot(b).T
    sub = M.dot(b)
    return A - (sub + sub.T)/2, M, V, d - A.shape[-1] - V.shape[-1] - 1

def moment_to_nat(params):
    S, M, V, nu = params
    b = V.dot(M.T)
    add = M.dot(b)
    return S + (add + add.T)/2, b, V, nu + S.shape[-1] + V.shape[-1] + 1

def expected_stats(natparam):
    S, M, V, nu = nat_to_moment(natparam)
    p = S.shape[-1]
    p2 = V.shape[-1]
    S_inv = inv_pd(S)
    const = log(2) * p
    E_Lambda = nu * S_inv
    E_Lambda_X = E_Lambda.dot(M)
    E_XT_Lambda_X = p * inv_pd(V) + M.T.dot(E_Lambda_X)
#     E_XT_Lambda_X = p * diag(1/diag(V)) + M.T.dot(E_Lambda_X)
    E_logdet_Lambda = digamma((nu - arange(p))/2).sum() + p * log(2) - slogdet(S)[1]
    return -E_Lambda/2, E_Lambda_X.T, -E_XT_Lambda_X/2, E_logdet_Lambda/2

def expected_stats_moment(param):
    S, M, V, nu = param
    p = S.shape[-1]
    p2 = V.shape[-1]
    S_inv = inv_pd(S)
    const = log(2) * p
    E_Lambda = nu * S_inv
    E_Lambda_X = E_Lambda.dot(M)
    E_XT_Lambda_X = p * inv_pd(V) + M.T.dot(E_Lambda_X)
#     E_XT_Lambda_X = p * diag(1/diag(V)) + M.T.dot(E_Lambda_X)
    E_logdet_Lambda = digamma((nu - arange(p))/2).sum() + p * log(2) - slogdet(S)[1]
    return -E_Lambda/2, E_Lambda_X.T, -E_XT_Lambda_X/2, E_logdet_Lambda/2

def expected_stats_straightthrough_stable(natparam, param):
    es = stop_gradient(expected_stats_moment(param))
    zero = tree_map(lambda x: x - stop_gradient(x), natparam)
    return tree_map(lambda x,y: x+y, es, zero)

def logZ(natparam):
    S, M, V, nu = nat_to_moment(natparam)
    p = S.shape[-1]
    p2 = V.shape[-1]
    const1 = log(2) * p
    const2 = log(2 * pi) * p * p2/2
    return nu/2 * (const1 - slogdet(S)[1]) + multigammaln(nu/2, p) - p/2 * slogdet(V)[1] + const2

def logZ_moment(param):
    S, M, V, nu = param
    p = S.shape[-1]
    p2 = V.shape[-1]
    const1 = log(2) * p
    const2 = log(2 * pi) * p * p2/2
    return nu/2 * (const1 - slogdet(S)[1]) + multigammaln(nu/2, p) - p/2 * slogdet(V)[1] + const2

def sample(natparam, key, n=1):
    S, M, V, nu = nat_to_moment(natparam)
    precision = tfd.WishartTriL(df=nu, scale_tril=cholesky(inv_pd(S))).sample(n, seed=key)
    output_shape = (-1,) + M.shape
    col_covar = inv_pd(precision)
    row_covar = inv_pd(V)
    return precision, tfd.MultivariateNormalFullCovariance(M.flatten(), kron(col_covar, row_covar)).sample(seed=key).reshape(output_shape)

def sample_es(natparam, key, n=1):
    precision, X = sample(natparam, key, n)
    neg_half_lam = -1/2 * precision
    log_det_lam = slogdet(precision)[1]/2
    Xt_lam = vmap(matmul)(X.swapaxes(-1,-2), precision)
    Xt_lam_X = -1/2 * vmap(dot)(Xt_lam, X)
    return neg_half_lam, Xt_lam, Xt_lam_X, log_det_lam

#unconstrained to natural
def uton(params):
    S_p, M, V_p, nu_p = params

    # construct positive definite matrix by softplus-ing diagonal
    latent_D = S_p.shape[-1]
    S = pd_param(S_p) + identity(latent_D) * 1e-3

    # do it again
    dim_V = V_p.shape[-1]
    V = pd_param(V_p) + identity(dim_V) * 1e-3
#     V = diag(softplus(diag(V_p))) + identity(dim_V) * 1e-3

    nu = softplus(nu_p) + latent_D - 1
    return moment_to_nat((S, M, V, nu))

# natural to unconstrained
def ntou(natparam):
    S, M, V, nu = nat_to_moment(natparam)

    latent_D = S.shape[-1]
    S_p = pd_param_inv(S - identity(latent_D) * 1e-3)

    dim_V = V.shape[-1]
    V_p = pd_param_inv(V - identity(dim_V) * 1e-3)
#     V_p = diag(softminus(diag(V - identity(dim_V) * 1e-3)))

    nu_p = softminus(nu - latent_D + 1)
    return S_p, M, V_p, nu_p

uton_natgrad = custom_vjp(uton)

def uton_natgrad_fwd(params):
    out = uton(params)
    return out, out

def uton_natgrad_bwd(resids, grads):
    return (jvp(ntou, (resids,), (grads,))[1],)

uton_natgrad.defvjp(uton_natgrad_fwd, uton_natgrad_bwd)