import jax
import jax.numpy as jnp                # JAX NumPy
from jax import custom_vjp
import numpy as np                     # Ordinary NumPy
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

from functools import partial
from jax.config import config 
from jax.scipy.special import logsumexp
import flax.linen as nn

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class ComplexNormal(object):
    def __init__(self, param):
        self.param = param

    def __repr__(self):
        return "ComplexNormal(param={})".format(self.param)

    def tree_flatten(self):
        children = (self.param,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    def mean(self):
        return jnp.exp(self.param)
    
    def log_prob(self, x):
        y = jnp.exp(self.param) + 1e-10
        return -(x/y - jnp.log(x/y) - 1.)
    
    @classmethod
    def parameter_properties(cls):
        class properties:
            is_preferred = True
            default_constraining_bijector_fn = lambda y: (lambda x: x)
            
        return dict(param=properties())
    
    
@register_pytree_node_class
class MPJPENormal(object):
    def __init__(self, param):
        self.param = param

    def __repr__(self):
        return "MPJPENormal(param={})".format(self.param)

    def tree_flatten(self):
        children = (self.param,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
    def mean(self):
        return self.param * 1000
    
    def log_prob(self, x):
        bs, seq_len, _ = x.shape
        x = x.reshape((bs, seq_len, -1, 3)) * 1000
        y = self.param.reshape((bs, seq_len, -1, 3)) * 1000
        ret = jnp.linalg.norm(x-y, axis=-1).mean(axis=-1, keepdims=True)
        return -ret
    
    @classmethod
    def parameter_properties(cls):
        class properties:
            is_preferred = True
            default_constraining_bijector_fn = lambda y: (lambda x: x)
            
        return dict(param=properties())
