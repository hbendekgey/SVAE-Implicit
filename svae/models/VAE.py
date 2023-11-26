from flax.linen import Module, Dense, BatchNorm, leaky_relu, softplus, compact
from jax.numpy import expand_dims, diag, zeros_like, ones_like
from jax import vmap
import jax
import jax.numpy as jnp
from typing import Callable, Any, Optional
ModuleDef = Any
from distributions import normal
from tensorflow_probability.substrates.jax import distributions as tfd
from networks.encoders import Encoder
from networks.decoders import SigmaDecoder, Decoder
from networks.sequence import ReverseLSTM, LSTM
import flax.linen as nn
from functools import partial

class VAE(Module):
    latent_D: int
    log_input: bool = False
    encoder_cls: ModuleDef = Encoder
    decoder_cls: ModuleDef = SigmaDecoder
    log_input: bool = False
    input_D: int = -1

    @compact
    def __call__(self, x, eval_mode=False, mask=None, theta_rng=None):
        if self.log_input:
            x = jnp.log(x)
        x_input = jnp.where(jnp.expand_dims(mask, -1), x, jnp.zeros_like(x)) if mask is not None else x
        natparam = self.encoder_cls(self.latent_D, name="encoder")(x, eval_mode=eval_mode, mask=mask)
        if x.ndim == 3:
            mu, var = vmap(vmap(normal.nat_to_moment))(natparam)
            q_z = tfd.Normal(mu.squeeze(-1), jnp.sqrt(vmap(vmap(diag))(var)))
        else:
            mu, var = vmap(normal.nat_to_moment)(natparam)
            q_z = tfd.Normal(mu.squeeze(-1), jnp.sqrt(vmap(diag)(var)))

        prior = tfd.Normal(zeros_like(q_z.loc), ones_like(q_z.loc))
        local_kl = tfd.kl_divergence(q_z, prior).sum()

        z_rng = self.make_rng('sampler')
        z = q_z.sample(seed=z_rng)
        if self.input_D == -1:
            likelihood = self.decoder_cls(x.shape[-1], name="decoder")(z, eval_mode=eval_mode)
        else:
            likelihood = self.decoder_cls(self.input_D, name="decoder")(z, eval_mode=eval_mode)
        if eval_mode:
            return likelihood, zeros_like(local_kl), local_kl, q_z
        else:
            return likelihood, zeros_like(local_kl), local_kl, z
        
class NonlinearDZ(Module):
    latent_D: int = 10
    network_width: int = 0
    
    @compact
    def __call__(self, z):
        network_width = max(self.network_width, self.latent_D)
        nu = nn.relu(nn.Dense(network_width)(z))
        nu = nn.sigmoid(nn.Dense(self.latent_D)(nu))
        
        mun = nn.relu(nn.Dense(network_width)(z))
        mun = nn.Dense(self.latent_D)(mun)
        
        mul = nn.Dense(self.latent_D)(z)
        
        mu = (1. - nu) * mul + nu * mun
        sigma = nn.softplus(nn.Dense(self.latent_D)(nn.relu(mun)))
        return tfd.Normal(mu, sigma)
    
    
class LinearDZ(Module):
    latent_D: int = 10
    network_width: int = 0
    
    @compact
    def __call__(self, z):
        mu = nn.Dense(self.latent_D)(z)
        sigma = nn.softplus(nn.Dense(self.latent_D)(0. * z))
        return tfd.Normal(mu, sigma)
    
    
class DKFCell(Module):
    latent_D: int = 10
    dz_cls: ModuleDef = NonlinearDZ
    gt_size: int = 64
    ez_size: int = 32
    
    @compact
    def __call__(self, sample, t_in):
        gt = nn.Dense(self.gt_size, name='gt')
        ez = nn.Dense(self.ez_size, name='ez')
        dz = self.dz_cls(self.latent_D, name='dz')
        mu, sigma = nn.Dense(self.latent_D, name='mu'), nn.Dense(self.latent_D, name='sigma')
        
        xt, eps, mask = t_in            
        xt = 0.5 * (jnp.tanh(gt(sample)) + xt)
        xt = jnp.tanh(ez(xt))
        q_mu, q_sigma = mu(xt), nn.softplus(sigma(xt))
        qz = tfd.Normal(q_mu, q_sigma)
        pz = dz(sample)
        kl = tfd.kl_divergence(qz, pz)
        sample = q_sigma * eps + q_mu
        pz_sample = jnp.clip(pz.scale * eps + pz.loc, -1e6, 1e6)
        sample = jnp.where(jnp.expand_dims(mask, axis=-1), sample, pz_sample)
        iwae_kl = (qz.log_prob(sample) - pz.log_prob(sample)).sum(axis=-1)
        return sample, (sample, kl, iwae_kl)
    
class DKFLatent(Module):
    latent_D: int = 10
    dz_cls: ModuleDef = NonlinearDZ
    gt_size: int = 64
    ez_size: int = 32
    
    @compact
    def __call__(self, inputs):
        xs, eps, mask = inputs
        init = jnp.zeros(eps.shape[:-2] + (self.latent_D,))
        cell = partial(DKFCell, dz_cls=self.dz_cls, latent_D=self.latent_D, gt_size=self.gt_size, ez_size=self.ez_size)
        ax = eps.ndim - 2
        S = nn.scan(cell, variable_broadcast="params", in_axes=ax, out_axes=ax,
                   split_rngs={"params": False})
        return S()(init, (xs, eps, mask))
    
class DKF(Module):
    latent_D: int = 10
    network_size: int = 64
    lstm_size: int = 64
    dz_cls: ModuleDef = NonlinearDZ
    decoder_cls: ModuleDef = Decoder
    encoder_network_cls: Optional[ModuleDef] = None
    gt_size: int = 64
    ez_size: int = 32
    log_input: bool = False
    
    @compact
    def __call__(self, x, eval_mode=False, mask=None, masking_mode=2, n_iwae_samples=0, theta_rng=None):
        z_rng = self.make_rng('sampler')
        if self.log_input:
            x = jnp.log(x)
        
        lengths = None
        if mask is None:
            mask = jnp.ones(x.shape[:2], dtype=jnp.int32)
        elif masking_mode == 0:
            lengths = mask.astype(jnp.int32).sum(axis=1)
            
        x = jnp.where(jnp.expand_dims(mask, -1), x, jnp.zeros_like(x))
        if masking_mode == 1:
            mask = jnp.ones_like(mask)
        
        if self.encoder_network_cls is None:
            xa = jnp.tanh(nn.Dense(self.network_size)(x))
            xa = ReverseLSTM(self.lstm_size)(xa, lengths, mask)
        else:
            xa = self.encoder_network_cls(self.lstm_size, name="encoder", eval_mode=eval_mode)(x, mask=mask)

        scanner = partial(DKFLatent, dz_cls=self.dz_cls, latent_D=self.latent_D, gt_size=self.gt_size, ez_size=self.ez_size, name='pgm')
        
        #if x.ndim == 3:
        #    scanner = nn.vmap(scanner)
        
        if n_iwae_samples > 0:
            eps_shape = xa.shape[:1] + (n_iwae_samples,) + xa.shape[1:-1] + (self.latent_D,)
            xa, mask = jnp.expand_dims(xa, 1), jnp.expand_dims(mask, 1)
        else:
            eps_shape = xa.shape[:-1] + (self.latent_D,)
        
        epsilon = tfd.Normal(jnp.zeros(eps_shape), jnp.ones(eps_shape)).sample(seed=z_rng)
        _, (z_sample, local_kl, iwae_kl) = scanner()((xa, epsilon, mask))
        
        local_kl_full = local_kl        
        likelihood = self.decoder_cls(x.shape[-1], name="decoder")(z_sample, eval_mode=eval_mode)
        local_kl = jnp.sum(local_kl) if n_iwae_samples == 0 else iwae_kl.sum(axis=-1)
                                            
        if eval_mode:
            return likelihood, zeros_like(local_kl), local_kl, (z_sample, local_kl_full)
        else:
            return likelihood, zeros_like(local_kl), local_kl, z_sample
        
        
        
class SRNNCell(Module):
    latent_D: int = 10
    dz_cls: ModuleDef = NonlinearDZ
    ez_cls: ModuleDef = NonlinearDZ
    
    @compact
    def __call__(self, sample, t_in):
        ez = self.ez_cls(self.latent_D, name='ez')
        dz = self.dz_cls(self.latent_D, name='dz')
        
        gt, ht, eps, mask, fixed_sample = t_in
        qz = ez(jnp.concatenate([sample, gt], axis=-1))
        pz = dz(jnp.concatenate([sample, ht], axis=-1))
        kl = tfd.kl_divergence(qz, pz)
        
        sample = qz.scale * eps + qz.loc
        pz_sample = jnp.clip(pz.scale * eps + pz.loc, -1e6, 1e6)
        sample = jnp.where(jnp.expand_dims(mask, axis=-1) == 1, sample, pz_sample)
        sample = jnp.where(jnp.expand_dims(mask, axis=-1) == 2, fixed_sample, sample)
        iwae_kl = (qz.log_prob(sample) - pz.log_prob(sample)).sum(axis=-1)
        return sample, (sample, kl, iwae_kl, qz.mean(), qz.stddev(), pz.mean(), pz.stddev())
    
class SRNNLatent(Module):
    latent_D: int = 10
    dz_cls: ModuleDef = NonlinearDZ
    ez_cls: ModuleDef = NonlinearDZ
    
    @compact
    def __call__(self, inputs):
        gs, hs, eps, mask, fixed_sample = inputs
        init = jnp.zeros(eps.shape[:-2] + (self.latent_D,))
        cell = partial(SRNNCell, dz_cls=self.dz_cls, ez_cls=self.ez_cls, latent_D=self.latent_D)
        ax = eps.ndim - 2
        S = nn.scan(cell, variable_broadcast="params", in_axes=ax, out_axes=ax,
                   split_rngs={"params": False})
        return S()(init, (gs, hs, eps, mask, fixed_sample))
    
    
class SRNN(Module):
    latent_D: int = 10
    network_size: int = 64
    lstm_size: int = 64
    log_input: bool = False
    dz_cls: ModuleDef = NonlinearDZ
    ez_cls: ModuleDef = NonlinearDZ
    decoder_cls: ModuleDef = Decoder
    encoder_network_cls: Optional[ModuleDef] = None
    
    @compact
    def __call__(self, x, eval_mode=False, mask=None, masking_mode=2, n_iwae_samples=0, theta_rng=None, fixed_samples = None):
        if self.log_input:
            x = jnp.log(x)
        z_rng = self.make_rng('sampler')
        
        lengths = None
        if mask is None:
            mask = jnp.ones(x.shape[:2], dtype=jnp.int32)
        elif masking_mode == 0:
            lengths = mask.astype(jnp.int32).sum(axis=1)
            
        x = jnp.where(jnp.expand_dims(mask, -1), x, jnp.zeros_like(x))
        if masking_mode == 1:
            mask = jnp.ones_like(mask)
            
        if fixed_samples is None:
            fixed_samples = jnp.zeros(x.shape[:2] + (self.latent_D,))

        xa = self.encoder_network_cls(self.lstm_size, name="encoder", eval_mode=eval_mode)(x, mask=mask)
        xa_shift = jnp.concatenate([jnp.zeros_like(xa[..., :1, :]), xa], axis=-2)[..., :-1, :]
        mask_shift = jnp.concatenate([jnp.ones_like(mask[..., :1]), mask], axis=-1)[..., :-1]
        ht = LSTM(self.lstm_size, name='ht')(xa_shift, lengths, mask_shift)
        gt = ReverseLSTM(self.lstm_size, name='gt')(jnp.concatenate([xa, ht], axis=-1), lengths, mask)

        scanner = partial(SRNNLatent, dz_cls=self.dz_cls, ez_cls=self.ez_cls, latent_D=self.latent_D,  name='pgm')
        
        #if x.ndim == 3:
        #    scanner = nn.vmap(scanner)
        
        if n_iwae_samples > 0:
            eps_shape = (n_iwae_samples,) + xa.shape[:1] + xa.shape[1:-1] + (self.latent_D,)
            #xa, mask = jnp.expand_dims(xa, 1), jnp.expand_dims(mask, 1)
            
            epsilon = tfd.Normal(jnp.zeros(eps_shape), jnp.ones(eps_shape)).sample(seed=z_rng)
            _, (z_sample, local_kl, iwae_kl, qz_mean, qz_std, pz_mean, pz_std) = jax.vmap(scanner(), ((None, None, 0, None),))((gt, ht, epsilon, mask, fixed_samples))
            z_sample, local_kl, iwae_kl = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), (z_sample, local_kl, iwae_kl))
            
            likelihood = self.decoder_cls(x.shape[-1], name="decoder")(jnp.concatenate([z_sample, jnp.repeat(jnp.expand_dims(ht, 1), n_iwae_samples, axis=1)], axis=-1), eval_mode=eval_mode)
        else:
            eps_shape = xa.shape[:-1] + (self.latent_D,)
        
            epsilon = tfd.Normal(jnp.zeros(eps_shape), jnp.ones(eps_shape)).sample(seed=z_rng)
            _, (z_sample, local_kl, iwae_kl, qz_mean, qz_std, pz_mean, pz_std) = scanner()((gt, ht, epsilon, mask, fixed_samples))
        
            likelihood = self.decoder_cls(x.shape[-1], name="decoder")(jnp.concatenate([z_sample, ht], axis=-1), eval_mode=eval_mode)
        local_kl = jnp.sum(local_kl) if n_iwae_samples == 0 else iwae_kl.sum(axis=-1)
                                            
        if eval_mode:
            return likelihood, zeros_like(local_kl), local_kl, (z_sample, qz_mean, qz_std, pz_mean, pz_std)
        else:
            return likelihood, zeros_like(local_kl), local_kl, (z_sample, qz_mean, qz_std, pz_mean, pz_std)
        
def eval_step_tf_impute(state, batch, sample_rng, mask=None, N_batches=1, **kwargs):
    mask = np.array(mask).astype(int)
    fill_mask = np.zeros_like(mask)
    fill_batch = np.array(batch)
    fixed_samples = np.zeros(batch.shape[:2] + (cfg.latent_D,))
    
    for step in tqdm.trange(batch.shape[-2]):
        fill_mask[:, :step] = 2 - mask[:, :step]
        fill_mask[:, step] = mask[:, step]
        state, loss, likelihood, aux = eval_step(state, fill_batch, mask=fill_mask, fixed_samples=fixed_samples, N_data=1, **kwargs)
        fixed_samples = aux['aux']

        new_rng, sample_rng = jax.random.split(sample_rng)
        sample = np.array(likelihood.sample(seed=new_rng))
        fill_batch[:, step] = batch[:, step] * mask[:, step, np.newaxis] + sample[:, step] * (1 - mask[:, step, np.newaxis])
    return fill_batch, state