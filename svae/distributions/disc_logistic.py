import jax
import jax.numpy as jnp                # JAX NumPy
from jax import custom_vjp
import numpy as np                     # Ordinary NumPy
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import parameter_properties

from functools import partial
from jax.config import config 
from jax.scipy.special import logsumexp
import flax.linen as nn

class DiscLogistic(tfd.Logistic):
    def log_prob(self, samples):
        log_scale = jnp.log(self.scale)
        centered = samples - self.loc                                         
        inv_stdv = jnp.exp(-log_scale)
        plus_in = inv_stdv * (centered + 1. / 255.)
        cdf_plus = nn.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / 255.)
        cdf_min = nn.sigmoid(min_in)
        log_cdf_plus = plus_in - nn.softplus(plus_in)
        log_one_minus_cdf_min = - nn.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - log_scale - 2. * nn.softplus(mid_in)

        log_prob_mid_safe = jnp.where(cdf_delta > 1e-5,
                                        jnp.log(jnp.maximum(cdf_delta, 1e-10)),
                                        log_pdf_mid - np.log(127.5))
        # woow the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = jnp.where(samples < -0.999, log_cdf_plus, jnp.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))  
        return log_probs