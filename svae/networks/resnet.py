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

import tqdm
from functools import partial
from jax.config import config 
from jax.scipy.special import logsumexp

from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, Iterable, Any
from jax_resnet.common import ModuleDef, Sequential
from jax_resnet.splat import SplAtConv2d
from .layers import ConvTransposeFixed, ConvUpsampling, leaky_relu, LocalConv

Shape = Iterable[int]
Distribution = Any

InitFn = Callable[[Any, Iterable[int], Any], Any]


class ConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = nn.relu
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    is_last: bool = False
    groups: int = 1
    dtype: Any = jnp.float32
    kernel_init: InitFn = nn.initializers.kaiming_normal()
    bias_init: InitFn = nn.initializers.zeros

    conv_cls: ModuleDef = nn.Conv
    norm_cls: Optional[ModuleDef] = nn.BatchNorm
    eval_mode: bool = False

    force_conv_bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = self.conv_cls(
            self.n_filters,
            self.kernel_size,
            self.strides,
            dtype=self.dtype,
            use_bias=(not self.norm_cls or self.force_conv_bias),
            padding='SAME',
            feature_group_count=self.groups,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        
        if self.groups > 1 and self.is_last:
            x = self.conv_cls(
              self.n_filters,
              (1, 1),
              (1, 1),
              dtype=self.dtype,
              use_bias=(not self.norm_cls or self.force_conv_bias),
              padding='SAME',
              feature_group_count=1,
              kernel_init=self.kernel_init,
              bias_init=self.bias_init,
            )(x)
        
        if self.norm_cls:
            scale_init =  (nn.initializers.zeros
                          if self.is_last else nn.initializers.ones)
            x = self.norm_cls(use_running_average=self.eval_mode, scale_init=scale_init, dtype=self.dtype)(x)

        if not self.is_last:
            x = self.activation(x)
        return x
      
class ResNetSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x, out_shape):
        if x.shape != out_shape:
            x = self.conv_block_cls(out_shape[-1],
                                    kernel_size=(1, 1),
                                    strides=self.strides,
                                    groups=1, eval_mode=self.eval_mode,
                                    activation=lambda y: y)(x)
        return x


class ResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    groups: int = 16
    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls, eval_mode=self.eval_mode)
        y = self.conv_block_cls(self.n_hidden,
                                padding=[(1, 1), (1, 1)],
                                strides=self.strides,  eval_mode=self.eval_mode, groups=self.groups)(x)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)],
                                is_last=True, eval_mode=self.eval_mode, groups=self.groups)(y)
        return self.activation(y + skip_cls(self.strides)(x, y.shape))

class ResNet(nn.Module):
    n_outputs: int
    block_cls: ModuleDef = ResNetBlock
    stage_sizes: Sequence[int] = (4, 4, 4, 4)
    hidden_sizes: Sequence[int] = (32, 64, 128, 256)
    conv_cls: ModuleDef = nn.Conv
    groups: int = 1
    dtype: Any = jnp.float32
    activation: Callable = leaky_relu
    norm_cls: Optional[ModuleDef] = nn.BatchNorm
    conv_block_cls: ModuleDef = ConvBlock
    eval_mode: bool = False
    extra_batch_dim: bool = True
    avg_pool: bool = False

    @nn.compact
    def __call__(self, x, return_encoding=False, mask=None):
        if x.ndim < 4:
            x = jnp.expand_dims(x, 1)
        batchsize = x.shape[0]
        if self.extra_batch_dim:
            x = jnp.concatenate(x, axis=0)
            
        conv_block_cls = partial(self.conv_block_cls, conv_cls=self.conv_cls, norm_cls=self.norm_cls, activation=self.activation, dtype=self.dtype)
        block_cls = partial(self.block_cls, conv_block_cls=conv_block_cls, eval_mode=self.eval_mode, groups=self.groups)

        stage_sizes, hidden_sizes = self.stage_sizes, self.hidden_sizes
        x = nn.Conv(16, (1, 1), dtype=self.dtype)(x)

        layers = []
        for i, (hsize, n_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
            for b in range(n_blocks):
                strides = (1, 1) if i == 0 or b != 0 else (2, 2)
                x = block_cls(n_hidden=hsize, strides=strides)(x)

        if self.avg_pool:
            x = jnp.mean(x, axis=(-2, -3))
        else:
            x = jnp.reshape(x, x.shape[:-3] + (-1,))
            
        if return_encoding:
            return x
        
        x = nn.Dense(self.n_outputs)(x)
        
        if self.extra_batch_dim:
            x = jnp.stack(jnp.split(x, batchsize))
        return x


class ResNetTranspose(nn.Module):
    n_outputs: int
    block_cls: ModuleDef = ResNetBlock
    stage_sizes: Sequence[int] = (4, 4, 4, 4)
    hidden_sizes: Sequence[int] = (32, 64, 128, 256)
    output_shape: Sequence[int] = (32, 32)
    conv_cls: ModuleDef = ConvUpsampling
    groups: int = 1
    dtype: Any = jnp.float32
    activation: Callable = leaky_relu
    norm_cls: Optional[ModuleDef] = None
    conv_block_cls: ModuleDef = ConvBlock
    eval_mode: bool = False
    extra_batch_dim: bool = True

    @nn.compact
    def __call__(self, x, mask=None):
        batchsize = x.shape[0]
        if self.extra_batch_dim:
            x = jnp.concatenate(x, axis=0)
            
        conv_block_cls = partial(self.conv_block_cls, conv_cls=self.conv_cls, norm_cls=self.norm_cls, activation=self.activation, dtype=self.dtype)
        block_cls = partial(self.block_cls, conv_block_cls=conv_block_cls,  eval_mode=self.eval_mode, groups=self.groups)

        stage_sizes, hidden_sizes = self.stage_sizes[::-1], self.hidden_sizes[::-1]
        ds_factor = (2 ** (len(stage_sizes) - 1))
        output_shape = self.output_shape[0] // ds_factor, self.output_shape[1] // ds_factor
        x = nn.Dense(output_shape[0] * output_shape[1] * hidden_sizes[0])(x)
        x = jnp.reshape(x, x.shape[:-1] + output_shape + (hidden_sizes[0],))

        layers = []
        for i, (hsize, n_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
            for b in range(n_blocks):
                strides = (1, 1) if i == 0 or b != 0 else (2, 2)
                x = block_cls(n_hidden=hsize, strides=strides)(x)

        x = nn.Conv(self.n_outputs, (1, 1), dtype=self.dtype)(x)
        if self.extra_batch_dim:
            x = jnp.stack(jnp.split(x, batchsize))
        return x
  