import jax
import jax.numpy as jnp                # JAX NumPy
from jax import custom_vjp

import flax
from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels
from tensorflow_probability.python.internal import reparameterization
from dataclasses import field

import tqdm
from functools import partial
from jax.config import config 
from jax.scipy.special import logsumexp
import jax.scipy as jsp
from jax.tree_util import register_pytree_node_class

from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, Iterable, Any, Dict
from jax_resnet.common import ConvBlock, ModuleDef, Sequential
from jax_resnet.splat import SplAtConv2d
from jax_resnet.resnet import ResNetBottleneckBlock
from .layers import ConvTransposeFixed, ConvUpsampling
from .dense import DenseNet


Distribution = Any

class Decoder(nn.Module):
    n_outputs: Optional[int] = None
    network_cls: ModuleDef = partial(DenseNet, norm_cls=None)
    likelihood: Distribution = tfd.Normal
    likelihood_dims: Optional[Sequence[int]] = None
    fixed_params: Dict = field(default_factory=dict)

    @nn.compact
    def __call__(self, x, eval_mode=False):
        likelihood_dims = self.likelihood_dims
        if likelihood_dims is None:
            likelihood_dims = [self.n_outputs for p, v in self.likelihood.parameter_properties().items() if v.is_preferred and p not in self.fixed_params.keys()]

        z_params = self.network_cls(sum(likelihood_dims), eval_mode=eval_mode)(x)

        param_names = [p for p, v in self.likelihood.parameter_properties().items() if v.is_preferred and p not in self.fixed_params.keys()]
        lparam_constraints = [v.default_constraining_bijector_fn() for p, v in self.likelihood.parameter_properties().items() if v.is_preferred and p not in self.fixed_params.keys()]
        split_zs = jnp.split(z_params, len(likelihood_dims), -1)
        params = dict([(n, f(p)) for n, f, p in zip(param_names, lparam_constraints, split_zs)])

        fixed_params = dict([(p, jnp.ones_like(split_zs[0]) * v) for p, v in self.fixed_params.items()])
        params.update(fixed_params)

        return self.likelihood(**params)
    

class CalibratedNormal(tfd.Normal):
    def log_prob(self, x):
        mse = ((self.loc - x) ** 2).mean(axis=tuple(range(x.ndim - 1)))
        rmse = jax.lax.stop_gradient(jnp.maximum(jnp.sqrt(mse), 1e-3))
        dist = tfd.Normal(self.loc, rmse)
        lp = dist.log_prob(x)
        diff = (self.scale - rmse) ** 2
        return lp - 0.001 * (diff + jax.lax.stop_gradient(diff))


class SigmaDecoder(nn.Module):
    n_outputs: int = 3
    network_cls: ModuleDef = partial(DenseNet, norm_cls=None)
    likelihood: Distribution = tfd.Normal

    @nn.compact
    def __call__(self, x, eval_mode = False):
        dev = self.param('dev', nn.initializers.ones, # Initialization function
                            (self.n_outputs,), x.dtype)
        z_mean = self.network_cls(self.n_outputs, eval_mode=eval_mode)(x)
        if eval_mode:
            return tfd.Normal(z_mean, nn.softplus(dev))
        else:
            return self.likelihood(z_mean, nn.softplus(dev))
    
class FixedDecoder(nn.Module):
    n_outputs: int = 3
    network_cls: ModuleDef = partial(DenseNet, norm_cls=None)
    likelihood: Distribution = tfd.Normal

    @nn.compact
    def __call__(self, x, eval_mode=False):
        dev = self.param('dev', nn.initializers.ones, # Initialization function
                            (self.n_outputs,), x.dtype)
        z_mean = self.network_cls(self.n_outputs, eval_mode=eval_mode)(x)
        return self.likelihood(z_mean, nn.softplus(dev) * 0. + 1.)


    
class MixtureDecoder(nn.Module):
    n_outputs: int = 3
    n_components: int = 10
    network_cls: ModuleDef = partial(DenseNet, norm_cls=None)
    likelihood: Distribution = tfd.Normal

    @nn.compact
    def __call__(self, x, eval_mode = False):
        outputs = 2 * self.n_components * self.n_outputs + self.n_components
        params = self.network_cls(outputs, eval_mode=eval_mode)(x)
        logits, params = jnp.split(params, [self.n_components,], axis=-1)
        means, logscales = jnp.split(params, 2, axis=-1)
        means = jnp.reshape(means, means.shape[:-1] + (self.n_components, self.n_outputs))
        logscales = jnp.reshape(logscales, logscales.shape[:-1] + (self.n_components, self.n_outputs))
        components_distribution = tfd.Independent(self.likelihood(means, nn.softplus(logscales)), 1)
        mixture_distribution = tfd.Categorical(logits=logits)
        return tfd.MixtureSameFamily(mixture_distribution=mixture_distribution,
          components_distribution=components_distribution)
    
@register_pytree_node_class  
class VMVMF(object):
    def __init__(self, loc, vm_concentration, mean_direction, vmf_concentration):
        self.VM = tfd.VonMises(loc=loc, concentration=vm_concentration)
        self.VMF = tfd.VonMisesFisher(mean_direction=mean_direction, concentration=vmf_concentration)
        self.params = (loc, vm_concentration, mean_direction, vmf_concentration)

    def __repr__(self):
        return "VonMises_VonMisesFisher(param={})".format(self.params)
    
    def tree_flatten(self):
        children = self.params
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def mean(self):
        dirs = self.VMF.mean()
        lens = self.VM.mean()
        out = dirs * lens[..., jnp.newaxis]
        return out.reshape(out.shape[:-2] + (-1,))
    
    def log_prob(self, x):
        x = x.reshape(x.shape[:-1] + (-1, 3))
        lens = jnp.linalg.norm(x, axis=-1)
        dirs = x / lens[..., jnp.newaxis]
        return self.VM.log_prob(lens) + self.VMF.log_prob(dirs)        

class VMVMF_Decoder(nn.Module):
    n_outputs: Optional[int] = None
    network_cls: ModuleDef = partial(DenseNet, norm_cls=None)

    @nn.compact
    def __call__(self, x, eval_mode=False):
        # x is batch x timsteps x ?
        # we then construct batch x timesteps x ? x 3
        n = self.n_outputs
        m = self.n_outputs//3

        z_params = self.network_cls(n * 2, eval_mode=eval_mode)(x)        
        
        mean_direction = z_params[...,:n].reshape(x.shape[:-1] + (-1, 3))
        mean_direction = mean_direction/jnp.linalg.norm(mean_direction, axis=-1,keepdims=True)
        
        vmf_concentration = jax.nn.softplus(z_params[...,n:n+m])
        loc = z_params[...,n+m:n+2*m]
        vm_concentration  = jax.nn.softplus(z_params[...,n+2*m:n+3*m])

        return VMVMF(loc, vm_concentration, mean_direction, vmf_concentration)