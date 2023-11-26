import jax
import jax.numpy as jnp                # JAX NumPy
from jax import custom_vjp

import flax
from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers

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
from jax_resnet.common import ConvBlock, ModuleDef, Sequential
from jax_resnet.splat import SplAtConv2d
from jax_resnet.resnet import ResNetBottleneckBlock
from jax import lax
import flax

Shape = Iterable[int]
Distribution = Any
Array = Any
Dtype = Any
PRNGKey = Any

def leaky_relu(x):
    return jnp.where(x > 0, x, 0.1 * x)

default_kernel_init = flax.linen.initializers.lecun_normal()

class LocalConv(nn.Module):
    features: int
    kernel_size: Iterable[int]
    strides: Union[None, int, Iterable[int]] = 1
    padding: Union[str, Iterable[Tuple[int, int]]] = 'SAME'
    input_dilation: Union[None, int, Iterable[int]] = 1
    kernel_dilation: Union[None, int, Iterable[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    precision: Any = None
    kernel_init: Callable[[jax.random.PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[jax.random.PRNGKey, Shape, Dtype], Array] = flax.linen.initializers.zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)

        if isinstance(self.kernel_size, int):
            raise TypeError('The kernel size must be specified as a'
                          ' tuple/list of integers (eg.: [3, 3]).')
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(x):
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return x

        is_single_input = False
        if inputs.ndim == len(kernel_size) + 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        strides = maybe_broadcast(self.strides)  # self.strides or (1,) * (inputs.ndim - 2)
        in_features = inputs.shape[-1]
        image_shape = (inputs.shape[-3] // strides[0], inputs.shape[-2] // strides[1])
        kernel_shape = kernel_size + (
            in_features // self.feature_group_count,)
        kernel_features = np.array(kernel_shape).prod()
        kernel_shape = image_shape + (kernel_features, self.features)
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        kernel = jnp.asarray(kernel, self.dtype)

        padding_lax = self.padding
        y = lax.conv_general_dilated_local(
            inputs,
            kernel,
            strides,
            padding_lax,
            filter_shape=kernel_size,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            precision=self.precision)


        if is_single_input:
            y = jnp.squeeze(y, axis=0)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, image_shape + (self.features,), self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            y += jnp.reshape(bias, (1,) + bias.shape)
        return y


class ConvTransposeFixed(nn.Module):
    features: int
    kernel_size: Iterable[int]
    strides: Tuple[int, int] = (1, 1)
    padding: Union[str, Iterable[Tuple[int, int]]] = 'VALID'
    feature_group_count: int = 1
    use_bias: bool = True
    kernel_init: Callable[[Any, Shape, Any], Any] = nn.initializers.kaiming_normal()
    bias_init: Callable[[Any, Shape, Any], Any] = nn.initializers.zeros
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Any) -> Any:
        x = nn.ConvTranspose(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
          dtype=self.dtype,
        )(x)
        return x

class ConvUpsampling(nn.Module):
    features: int
    kernel_size: Iterable[int]
    strides: Tuple[int, int] = (1, 1)
    padding: Union[str, Iterable[Tuple[int, int]]] = 'VALID'
    feature_group_count: int = 1
    use_bias: bool = True
    kernel_init: Callable[[Any, Shape, Any], Any] = nn.initializers.kaiming_normal()
    bias_init: Callable[[Any, Shape, Any], Any] = nn.initializers.zeros
    conv_cls: Callable = nn.Conv
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Any) -> Any:
        if self.strides[0] + self.strides[1] > 2:
            x = jax.image.resize(x, x.shape[:-3] + (x.shape[-3] * self.strides[0], x.shape[-2] * self.strides[1], x.shape[-1]), "nearest")
        x = self.conv_cls(
            features=self.features,
            kernel_size=self.kernel_size,
            use_bias=self.use_bias,
            feature_group_count=self.feature_group_count,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype
        )(x)
        return x
    
class LayerNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.ones
    use_running_average: bool = True
    
    @nn.compact
    def __call__(self, x):
        ra_mean = self.variable('batch_stats', 'mean',
                            lambda s: jnp.zeros(s, jnp.float32),
                            (1,))
        
        return nn.LayerNorm(epsilon=self.epsilon, dtype=self.dtype,
                            param_dtype=self.param_dtype, use_bias=self.use_bias,
                            use_scale=self.use_scale, bias_init=self.bias_init,
                            scale_init=self.scale_init)(x)
    
    