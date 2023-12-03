from jax import custom_vjp, jvp
from jax.numpy import log
from jax.lax import lgamma
from jax.scipy.special import digamma
import tensorflow_probability.substrates.jax.distributions as tfd
from utils import softminus
from flax.linen import softplus
# Automatically batched. Assumes last dimension is the parameter dimension, with length K

def expected_stats(natparam):
    alpha = natparam + 1
    return digamma(alpha) - digamma(alpha.sum(axis=-1, keepdims=True))

def logZ(natparam):
    alpha = natparam + 1
    return lgamma(alpha).sum(axis=-1) - lgamma(alpha.sum(axis=-1))

def sample(natparam, key, n=1):
    return tfd.Dirichlet(natparam + 1).sample(n, seed=key)

def sample_es(natparam, key, n=1):
    return log(sample(natparam, key, n))

def uton(param):
    return softplus(param) - 0.99

def ntou(natparam):
    return softminus(natparam + 0.99)

uton_natgrad = custom_vjp(uton)

def uton_natgrad_fwd(params):
    return uton(params), uton(params)

def uton_natgrad_bwd(resids, grads):
    return (jvp(ntou, (resids,), (grads,))[1],)

uton_natgrad.defvjp(uton_natgrad_fwd, uton_natgrad_bwd)