from jax import jit, vmap
from jax.numpy import amax, zeros
from jax.nn import softmax, relu
from jax.scipy.special import logsumexp
from utils import straight_through
import tensorflow_probability.substrates.jax.distributions as tfd
# Automatically batched

expected_stats = jit(softmax)

@jit
def logZ(natparam):
    return logsumexp(natparam, axis=-1)

@jit
def expected_stats_underflow_stable(natparam):
    if natparam.dtype == "float64":
        min_log_prob = 24
    else:
        min_log_prob = 12
    def clip_log_probs(x):
        return relu(x - amax(x,-1,keepdims=True) + min_log_prob)
    return expected_stats(straight_through(clip_log_probs)(natparam))

def sample(natparam, key, n=1):
    return tfd.Categorical(logits=natparam).sample(n, seed=key)

def sample_es(natparam, key, n=1):
    samples = sample(natparam, key, n)
    sample_to_es = lambda x: zeros(natparam.shape[-1]).at[x].set(1)
    return vmap(vmap(sample_to_es))(samples)
