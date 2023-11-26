from flax.linen import Module, Dense, BatchNorm, leaky_relu, softplus, compact
from jax.numpy import expand_dims, diag, zeros_like, ones_like
from jax import vmap
import jax.numpy as jnp
from typing import Callable, Optional, Any, Sequence
from distributions import normal
from tensorflow_probability.substrates.jax import distributions as tfd
from .dense import DenseNet
from functools import partial
import jax.numpy as jnp
import flax.linen as nn

ModuleDef = Any

class EncoderNatural(Module):
    latent_D: int
    groups: int = 1
    network_cls: ModuleDef = DenseNet
    skip_connection: bool = True
    loc_norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9)
    scale_norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9)

    @nn.compact
    def __call__(self, x, eval_mode = False, mask=None):
        z_params = self.network_cls(2 * self.latent_D, eval_mode=eval_mode)(x, mask=mask)
        loc, scale = jnp.split(z_params, 2, axis=-1)
        if self.skip_connection:
            loc = loc + Dense(self.latent_D)(x)
        if self.loc_norm_cls:
            loc = self.loc_norm_cls(use_running_average=eval_mode)(loc)
        loc = jnp.reshape(loc, loc.shape[:-1] + (loc.shape[-1] // self.groups, self.groups))
        
        if self.scale_norm_cls:
            scale = self.scale_norm_cls(use_running_average=eval_mode)(scale)
        scale = softplus(scale)
        
        all_scale = []
        for s in jnp.split(scale, self.groups, axis=-1):
            if x.ndim == 3:
                all_scale.append(-vmap(vmap(diag))(s))
            else:
                all_scale.append(-vmap(diag)(s))
        scale = jnp.concatenate(all_scale, axis=-1)
        return scale, loc
    
class Encoder(Module):
    latent_D: int
    groups: int = 1
    network_cls: ModuleDef = DenseNet
    skip_connection: bool = False
    loc_norm_cls: Optional[ModuleDef] = None #partial(nn.BatchNorm, momentum=0.9)
    scale_norm_cls: Optional[ModuleDef] = None #partial(nn.BatchNorm, momentum=0.9)
    preserve_compat: bool = True

    @nn.compact
    def __call__(self, x, eval_mode = False, mask=None):
        z_params = self.network_cls(2 * self.latent_D, eval_mode=eval_mode)(x, mask=mask)
        loc, inv_scale = jnp.split(z_params, 2, axis=-1)
        if self.skip_connection:
            loc = loc + Dense(self.latent_D, dtype=loc.dtype)(x)
        if self.loc_norm_cls:
            loc = self.loc_norm_cls(use_running_average=eval_mode)(loc)
        
        if self.scale_norm_cls:
            inv_scale = self.scale_norm_cls(use_running_average=eval_mode)(inv_scale)
        loc = jnp.exp(inv_scale) * loc
        loc = jnp.reshape(loc, loc.shape[:-1] + (loc.shape[-1] // self.groups, self.groups))

        inv_scale = 0.5 * jnp.exp(inv_scale) if self.preserve_compat else jnp.exp(inv_scale)
        
        all_scale = []
        for s in jnp.split(inv_scale, self.groups, axis=-1):
            if x.ndim == 3:
                all_scale.append(-1/2 * vmap(vmap(diag))(s))
            else:
                all_scale.append(-1/2 * vmap(diag)(s))
        inv_scale = jnp.concatenate(all_scale, axis=-1)
        return inv_scale, loc
    
class SigmaEncoder(Module):
    latent_D: int
    groups: int = 1
    network_cls: ModuleDef = DenseNet
    skip_connection: bool = False
    loc_norm_cls: Optional[ModuleDef] = None #partial(nn.BatchNorm, momentum=0.9)
    scale_norm_cls: Optional[ModuleDef] = None #partial(nn.BatchNorm, momentum=0.9)
    preserve_compat: bool = True

    @nn.compact
    def __call__(self, x, eval_mode = False, mask=None):
        loc = self.network_cls(self.latent_D, eval_mode=eval_mode)(x, mask=mask)
        inv_scale = self.param('scale', nn.initializers.normal(), loc.shape[-1:], loc.dtype)
        inv_scale = inv_scale.reshape([1] * (len(loc.shape) - 1) + [-1]) * jnp.ones_like(loc)
        
        if self.skip_connection:
            loc = loc + Dense(self.latent_D)(x)
        if self.loc_norm_cls:
            loc = self.loc_norm_cls(use_running_average=eval_mode)(loc)
        
        if self.scale_norm_cls:
            inv_scale = self.scale_norm_cls(use_running_average=eval_mode)(inv_scale)
        loc = jnp.exp(inv_scale) * loc
        loc = jnp.reshape(loc, loc.shape[:-1] + (loc.shape[-1] // self.groups, self.groups))
        inv_scale = 0.5 * jnp.exp(inv_scale) if self.preserve_compat else jnp.exp(inv_scale)
        
        all_scale = []
        for s in jnp.split(inv_scale, self.groups, axis=-1):
            if x.ndim == 3:
                all_scale.append(-1/2 * vmap(vmap(diag))(s))
            else:
                all_scale.append(-1/2 * vmap(diag)(s))
        inv_scale = jnp.concatenate(all_scale, axis=-1)
        return inv_scale, loc
    
class SigmaEncoderAdapt(Module):
    latent_D: int
    groups: int = 1
    network_cls: ModuleDef = DenseNet
    skip_connection: bool = False
    preserve_compat: bool = True

    @nn.compact
    def __call__(self, x, eval_mode = False, mask=None):
        params = self.network_cls(self.latent_D * 2, eval_mode=eval_mode)(x, mask=mask)
        loc, _ = jnp.split(params, 2, -1)

        inv_scale = self.param('scale', nn.initializers.normal(), loc.shape[-1:], loc.dtype)
        inv_scale = inv_scale.reshape([1] * (len(loc.shape) - 1) + [-1]) * jnp.ones_like(loc)
        
        loc = jnp.exp(inv_scale) * loc
        loc = jnp.reshape(loc, loc.shape[:-1] + (loc.shape[-1] // self.groups, self.groups))
        inv_scale = 0.5 * jnp.exp(inv_scale) if self.preserve_compat else jnp.exp(inv_scale)
        
        all_scale = []
        for s in jnp.split(inv_scale, self.groups, axis=-1):
            if x.ndim == 3:
                all_scale.append(-1/2 * vmap(vmap(diag))(s))
            else:
                all_scale.append(-1/2 * vmap(diag)(s))
        inv_scale = jnp.concatenate(all_scale, axis=-1)
        return inv_scale, loc
    
    