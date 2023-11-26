from networks.dense import DenseBlock
import flax.linen as nn
import numpy as np
from flax.linen import Module, Dense, LayerNorm, gelu, softplus, compact
from jax.numpy import expand_dims, diag, zeros_like, ones_like
from jax import vmap
from typing import Callable, Optional, Any, Sequence
from distributions import normal
from functools import partial
import jax.numpy as jnp
import jax
from tensorflow_probability.substrates.jax import distributions as tfd
from networks.sequence import SimpleLSTM, SimpleBiLSTM, ReverseLSTM
ModuleDef = Any
import math
from networks.layers import LayerNorm
from networks.decoders import Decoder, SigmaDecoder

class PositionalEncoding(nn.Module):
    d_model : int        # Hidden dimensionality of the input.
    max_len : int = 1000  # Maximum length of a sequence to expect.

    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len, dtype=np.float32)[:,None]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

    def __call__(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x

class TransformerEncoder(nn.Module):
    n_outputs: int
    d_model: int = 256
    dtype: Any = jnp.float32
    activation: Callable = gelu
    norm_cls: Optional[ModuleDef] = LayerNorm
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x, mask=None):
        # Linear
        x = self.activation(nn.Dense(self.d_model, dtype=self.dtype)(x))
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
            
        # Positional Encoding
        x = PositionalEncoding(self.d_model)(x)

        # Single head attention
        attn_mask = None if mask is None else jnp.repeat(jnp.expand_dims(mask, (1, 2)), mask.shape[1], 2)
        x = nn.SelfAttention(1)(x, mask=attn_mask) + x
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)

        # Linear again
        x = self.activation(nn.Dense(self.d_model, dtype=self.dtype)(x)) + x
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
        
        x = nn.Dense(self.n_outputs, dtype=self.dtype)(x)
        return x

class TransformerDecoder(nn.Module):
    n_outputs: int
    d_model: int = 256
    decoder_cls: ModuleDef = SigmaDecoder
    dtype: Any = jnp.float32
    activation: Callable = gelu
    norm_cls: Optional[ModuleDef] = LayerNorm
    eval_mode: bool = False

    @nn.compact
    def __call__(self, x, z, self_mask=None, cross_mask=None):
        # Linear
        z = self.activation(nn.Dense(self.d_model, dtype=self.dtype)(z))
        if self.norm_cls:
            z = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(z)
            
        # Positional Encoding
        z = PositionalEncoding(self.d_model)(z)

        # Causal Single head attention
        causal_mask = np.tril(np.ones((z.shape[1], z.shape[1])))
        if not self_mask is None:
            causal_mask = jnp.expand_dims(causal_mask, (0, 1)) * jnp.expand_dims(self_mask, (1, 2))
        z = nn.SelfAttention(1, dtype=self.dtype)(z, mask=causal_mask) + z
        if self.norm_cls:
            z = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(z)

        # Now, cross attention
        if z.shape[1] > x.shape[1]:
            causal_mask = np.tril(np.ones((z.shape[1], x.shape[1])), -1)
        else:
            causal_mask = np.tril(np.ones((z.shape[1], x.shape[1])))
        if not cross_mask is None:
            causal_mask = jnp.expand_dims(causal_mask, (0, 1)) * jnp.expand_dims(cross_mask, (1, 2))
        x = nn.MultiHeadDotProductAttention(1, dtype=self.dtype)(z, x, mask=causal_mask)
        if z.shape[1] > x.shape[1]:
            x = x.at[...,0,:].set(0)
        x = x + z
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)

        # Linear again
        x = self.activation(nn.Dense(self.d_model, dtype=self.dtype)(x)) + x
        if self.norm_cls:
            x = self.norm_cls(use_running_average=self.eval_mode, dtype=self.dtype)(x)
        
        likelihood = self.decoder_cls(self.n_outputs, network_cls = lambda x, eval_mode: nn.Dense(x) )(x, eval_mode=self.eval_mode)
        return likelihood
    
class LigHTVAE(nn.Module):
    latent_D: int
    d_model: int = 256
    decoder_cls: ModuleDef = SigmaDecoder
    dtype: Any = jnp.float32
    activation: Callable = gelu
    norm_cls: Optional[ModuleDef] = LayerNorm
    log_input: bool = False

    @nn.compact
    def __call__(self, x, mask=None, eval_mode=False, fixed_samples=None):
        if self.log_input:
            x = jnp.log(x)

        if not (mask is None):
            unscaled_mask = mask
            mask = jnp.where(mask > 0, jnp.ones_like(mask), jnp.zeros_like(mask))
            
        encoder = TransformerEncoder(self.latent_D * 2, d_model=self.d_model, 
                                    dtype=self.dtype, activation = self.activation,
                                    norm_cls = self.norm_cls, eval_mode=eval_mode, name='encoder')
        params = encoder(x, mask=mask)
        mu, var = jnp.split(params, 2, -1)
        q_z = tfd.Normal(mu, softplus(var))

        z_rng = self.make_rng('sampler')
        z = q_z.sample(seed=z_rng)
        
        if fixed_samples is None:
            fixed_samples = jnp.zeros(x.shape[:2] + (self.latent_D,))
        if not (mask is None):
            z = jnp.where(jnp.expand_dims(unscaled_mask, axis=-1) == 1, z, jnp.zeros_like(z))
            z = jnp.where(jnp.expand_dims(unscaled_mask, axis=-1) == 2, fixed_samples, z)
        decoder_mask = None if mask is None else mask[..., :-1]

        z_decoder = TransformerDecoder(self.latent_D, d_model=self.d_model,
                                       decoder_cls = partial(Decoder, likelihood=tfd.Normal),
                                       dtype=self.dtype, activation = self.activation,
                                       norm_cls = self.norm_cls, eval_mode=eval_mode, name='pgm')

        x_decoder = TransformerDecoder(x.shape[-1], d_model=self.d_model, 
                                       decoder_cls = self.decoder_cls, dtype=self.dtype, 
                                       activation = self.activation, norm_cls = self.norm_cls, 
                                       eval_mode=eval_mode, name='decoder')
        
        p_z2 = z_decoder(x[...,:-1,:], z[...,:-1,:], self_mask=decoder_mask, cross_mask=decoder_mask)
        loc, scale = p_z2.loc, p_z2.scale
        full_loc = jnp.concatenate([jnp.zeros_like(loc[...,:1,:]), loc], -2)
        full_scale = jnp.concatenate([jnp.ones_like(scale[...,:1,:]), scale], -2)
        p_z = tfd.Normal(loc = full_loc, scale = full_scale)
        
        self_decoder_mask = None
        if not mask is None:
            z = jnp.where(jnp.expand_dims(mask, axis=-1) == 0, p_z.sample(seed=z_rng), z)
            self_decoder_mask = jnp.concatenate([jnp.ones_like(mask[...,:1]), decoder_mask], -1)
            
        p_x = x_decoder(x[...,:-1,:], z, self_mask=self_decoder_mask, cross_mask=decoder_mask)
        
        local_kl = tfd.kl_divergence(q_z, p_z).sum()
        return p_x, zeros_like(local_kl), local_kl, z