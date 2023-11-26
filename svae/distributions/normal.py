from jax import jit, vmap
from jax.numpy import log, pi, tile, identity, zeros_like, where, expand_dims, outer
from jax.numpy.linalg import slogdet
from utils import solve_pd, inv_pd, T
import tensorflow_probability.substrates.jax.distributions as tfd
# Multivariate normal distribution
# with natural parameters
#    Sigma^-1 * mu (D x 1)
#    -1/2 Sigma^-1 (D x D)
#
# and sufficient statistics
#    x    (D x 1)
#    xT x (D x D)
#
# NOT automatically batched

@jit
def nat_to_mean_and_precision(natparam):
    J, h = natparam
    return solve_pd(-2 * J, h), -2 * J

@jit
def nat_to_moment(natparam):
    J, h = natparam
    var = inv_pd(-2 * J)
    mu = var.dot(h)
    return mu, var

# same as above, but where -1/2 factor on J has been removed
@jit
def Jh_to_moment(natparam):
    J, h = natparam
    var = inv_pd(J)
    mu = var.dot(h)
    return mu, var

@jit
def moment_to_nat(params):
    mu, var = params
    J = inv_pd(var)
    h = J.dot(mu)
    return -1/2 * J, h

@jit
def expected_stats(natparam):
    mu, var = nat_to_moment(natparam)
    return (var + mu * mu.T, mu)

@jit
def expected_stats_masked(potentials, default_var = 1e5):
    J, h = potentials
    J = J - identity(J.shape[-1], J.dtype)/(2 * default_var)
    return vmap(expected_stats)((J,h))

@jit
def logZ(natparam):
    J, h = natparam
    const = log(pi * 2) * J.shape[-1]
    return 1/2 * (h.T.dot(solve_pd(-2 *J,h)).squeeze() - slogdet(-2 * J)[1] + const)

def sample(natparam, key, n=1):
    mu, cov = nat_to_moment(natparam)
    return tfd.MultivariateNormalFullCovariance(mu.squeeze(-1), cov).sample(n, seed=key)

def sample_es(natparam, key, n=1):
    samples = sample(natparam, key, n)
    return vmap(outer)(samples, samples), expand_dims(samples, -1)

def sample_from_es(expected_stats, key, n=1):
    mu = expected_stats[1]
    cov = expected_stats[0] - mu * T(mu)
    return tfd.MultivariateNormalFullCovariance(mu.squeeze(-1), cov).sample(n, seed=key)

def sample_es_from_es(expected_stats, key, n=1):
    samples = sample_from_es(expected_stats, key, n)
    return vmap(outer)(samples, samples), expand_dims(samples, -1)