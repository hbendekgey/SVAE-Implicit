from flax.linen import Module, Dense, BatchNorm, leaky_relu, softplus, compact
from jax.numpy import expand_dims, diag, zeros_like, ones_like
from jax import vmap
import jax
import jax.numpy as jnp
from typing import Callable, Any, Dict, Optional
ModuleDef = Any
from distributions import normal
from tensorflow_probability.substrates.jax import distributions as tfd
from models.SVAE_SLDS import PGM_SLDS
from networks.encoders import Encoder
from networks.decoders import SigmaDecoder, Decoder
from networks.sequence import ReverseLSTM, LSTM
import flax.linen as nn
from dataclasses import field
from functools import partial
from distributions import mniw, niw, dirichlet, normal
from utils import inv_pd, solve_pd_stable, softminus, pd_param, pd_param_inv, straight_through
from jax.experimental.host_callback import id_print
from jax.scipy.special import logsumexp

class NonlinearEZ(Module):
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
    
    
class LinearEZ(Module):
    latent_D: int = 10
    network_width: int = 0
    
    @compact
    def __call__(self, z):
        mu = nn.Dense(self.latent_D)(z)
        sigma = nn.softplus(nn.Dense(self.latent_D)(0. * z))
        return tfd.Normal(mu, sigma)
    
class SmallTanhNet(Module):
    hidden_width: int = 64
    output_width: int = 64
    
    @compact
    def __call__(self, x):
        x = jnp.tanh(nn.Dense(self.hidden_width)(x))
        x = jnp.tanh(nn.Dense(self.hidden_width)(x))
        return nn.Dense(self.output_width)(x)
    
class SmallELUNet(Module):
    hidden_width: int = 64
    output_width: int = 64
    
    @compact
    def __call__(self, x):
        x = nn.gelu(nn.Dense(self.hidden_width)(nn.LayerNorm()(x)))
        x = nn.gelu(nn.Dense(self.hidden_width)(nn.LayerNorm()(x)))
        return nn.Dense(self.output_width)(nn.LayerNorm()(x))
    
    
class DSLDSCell(Module):
    latent_D: int = 10
    K: int = 50
    net_cls: ModuleDef = SmallELUNet
    fz_size: int = 64
    ez_size: int = 64
    gradient_estimator: str = 'concrete'
    pk_net_cls: Any = None
    pz_net_cls: Any = None
    
    @compact
    def __call__(self, carry, t_in):
        temp, rng, z_sample, k_sample, pgm_params = carry        
        (transition, start_pk, Ab, Q, start_mu, start_Q) = pgm_params
                
        k_rng, z_rng, rng = jax.random.split(rng, 3)
        # Sample the next state using the VI dist.
        xt, eps, mask = t_in 
        fz = self.net_cls(self.fz_size, self.K)
        
        z_sample_finite = jnp.where(jnp.isfinite(z_sample), z_sample, jnp.zeros_like(z_sample))
        k_sample_finite = jnp.where(jnp.isfinite(k_sample), k_sample, jnp.ones_like(k_sample))
        qk_log_gammat = fz(jnp.concatenate([z_sample_finite, k_sample_finite, xt], axis=-1))
        
        if self.gradient_estimator == 'concrete':
            qk = tfd.RelaxedOneHotCategorical(temp, logits=qk_log_gammat)
            qk_sample = qk.sample(seed=k_rng)
        elif self.gradient_estimator == 'straight_through':
            qk = tfd.OneHotCategorical(logits=qk_log_gammat)
            qk_sample = straight_through(lambda p: tfd.OneHotCategorical(probs=p).sample(seed=k_rng))(nn.softmax(qk_log_gammat))
        elif self.gradient_estimator == 'straight_through_logits':
            qk = tfd.OneHotCategorical(logits=qk_log_gammat)
            qk_sample = straight_through(lambda p: tfd.OneHotCategorical(logits=p).sample(seed=k_rng))(qk_log_gammat)
        else:
            raise Exception('Unknown gradient estimator for DSLDS!')
        
        # Get the normalized transition matrix and the next state distribution
        if self.pk_net_cls is None:
            pk_gammat = jnp.einsum('nk,nkj->nj', k_sample_finite, transition)
            log_pk_gammat = jnp.log(pk_gammat)
        else:
            pk_net = self.pk_net_cls(self.fz_size, self.K)
            log_pk_gammat = pk_net(jnp.concatenate([z_sample_finite, k_sample_finite], axis=-1))
                                   
        # Use a different state distribution for step 0
        log_pk_gammat = jnp.where(jnp.isfinite(k_sample), log_pk_gammat, jnp.log(start_pk))
        
        # Sample the next state using the prior
        if self.gradient_estimator == 'concrete':
            pk = tfd.RelaxedOneHotCategorical(temp, logits=log_pk_gammat)
            pk_sample = pk.sample(seed=k_rng)
        elif self.gradient_estimator == 'straight_through':
            pk = tfd.OneHotCategorical(logits=log_pk_gammat)
            pk_sample = straight_through(lambda p: tfd.OneHotCategorical(probs=p).sample(seed=k_rng))(nn.softmax(log_pk_gammat))
        elif self.gradient_estimator == 'straight_through_logits':
            pk = tfd.OneHotCategorical(logits=log_pk_gammat)
            pk_sample = straight_through(lambda p: tfd.OneHotCategorical(logits=p).sample(seed=k_rng))(log_pk_gammat)
        else:
            raise Exception('Unknown gradient estimator for DSLDS!')
            
        k_sample = jnp.where(jnp.expand_dims(mask, axis=-1), qk_sample, pk_sample)
        
        # KL-divergence for the discrete states (pretend 
        discrete_iwae_kl = (qk.log_prob(k_sample) - pk.log_prob(k_sample))
        if self.gradient_estimator == 'straight_through' or self.gradient_estimator == 'straight_through_logits':
            discrete_kl = tfd.kl_divergence(tfd.Categorical(logits=qk_log_gammat), tfd.Categorical(logits=log_pk_gammat))
        else:
            discrete_kl = discrete_iwae_kl
        #
        
        # Get the LDS params for the current state
        if self.pz_net_cls is None:
            A, b = Ab[..., :-1], Ab[..., -1]
            Ak = jnp.einsum('nk,nkde->nde', k_sample, A)
            bk = jnp.einsum('nk,nkd->nd', k_sample, b)
            pz_mu = jnp.einsum('nd,nde->ne', z_sample_finite, Ak) + bk
            pz_mu = jnp.where(jnp.isfinite(z_sample), pz_mu, start_mu)
            Qk = jnp.einsum('nk,nkde->nde', k_sample, Q)
            Qk = jnp.where(jnp.isfinite(jnp.expand_dims(z_sample, -1)), Qk, start_Q)
            pz = tfd.MultivariateNormalFullCovariance(pz_mu, Qk)
        else:
            pz_net = self.pz_net_cls(self.fz_size, self.latent_D)
            pz_mu = pz_net(jnp.concatenate([z_sample_finite, k_sample], axis=-1))
            pz_log_sigma = self.param('pz_sigma', lambda rng: 0.1 * jax.random.normal(rng, (1, self.latent_D), dtype=jnp.float32))
            pz_sigma = jnp.exp(jnp.maximum(pz_log_sigma, -3)) + 0 * pz_mu
            pz = tfd.MultivariateNormalDiag(pz_mu, pz_sigma)
            
        # Get the variational distribution for z
        W = self.param('W', lambda rng: 0.1 * jax.random.normal(rng, (self.K, self.latent_D, xt.shape[-1]), dtype=jnp.float32))
        c = self.param('c', lambda rng: 0.1 * jax.random.normal(rng, (self.K, xt.shape[-1]), dtype=jnp.float32))
        
        Wk = jnp.einsum('nk,kde->nde', k_sample, W)
        ck = jnp.einsum('nk,kd->nd', k_sample, c)
        gt = jnp.einsum('nd,nde->ne', z_sample_finite, Wk) + ck
        
        ez = self.net_cls(self.ez_size, 2 * self.latent_D)
        #qz_mu_log_sigma = ez(jnp.concatenate([jnp.tanh(gt), xt], axis=-1))
        qz_mu_log_sigma = ez(jnp.concatenate([gt, k_sample, xt], axis=-1))
        qz_mu, qz_log_sigma = jnp.split(qz_mu_log_sigma, 2, axis=-1)
        qz_log_sigma = self.param('qz_sigma', lambda rng: 0.1 * jax.random.normal(rng, (1, self.latent_D), dtype=jnp.float32))
        
        qz_sigma = jnp.exp(jnp.maximum(qz_log_sigma, -3)) + 0 * qz_mu
        qz = tfd.MultivariateNormalDiag(qz_mu, qz_sigma)
        
        #def combine_qz(qzm, qzs, pzm, pzs):
        #    qz_J, qz_h = normal.moment_to_nat((qzm, jnp.diag(qzs)))
        #    pz_J, pz_h = normal.moment_to_nat((pzm, pzs))
        #    return normal.nat_to_moment((qz_J + pz_J, qz_h + pz_h))
        #
        #qz_mu, qz_sigma = jax.vmap(combine_qz)(qz_mu, qz_sigma, pz_mu, Qk)
        #qz = tfd.MultivariateNormalFullCovariance(qz_mu, qz_sigma)

        # Get the kl divergence and sample
        kl = tfd.kl_divergence(qz, pz)
        sample = qz.sample(seed=z_rng)
        pz_sample = jnp.clip(pz.sample(seed=z_rng), -1e2, 1e2)
        #pz_sample = pz.mean()
        sample = jnp.where(jnp.expand_dims(mask, axis=-1), sample, pz_sample)
        iwae_kl = (qz.log_prob(sample) - pz.log_prob(sample)).sum(axis=-1)
        return (temp, rng, sample, k_sample, pgm_params), (sample, kl + discrete_kl, iwae_kl + discrete_iwae_kl, nn.softmax(qk_log_gammat))
    
class DSLDSLatent(Module):
    latent_D: int = 10
    K: int = 50
    net_cls: ModuleDef = SmallELUNet
    fz_size: int = 64
    ez_size: int = 64
    gradient_estimator: str = 'concrete'
    pk_net_cls: Any = None
    pz_net_cls: Any = None
    
    @compact
    def __call__(self, inputs):
        xs, eps, mask, temp, rng, pgm_params = inputs
        init = temp, rng, jnp.nan * jnp.zeros(eps.shape[:-2] + (self.latent_D,)), jnp.nan * jnp.zeros(eps.shape[:-2] + (self.K,)), pgm_params 
        cell = partial(DSLDSCell, latent_D=self.latent_D, K=self.K, fz_size=self.fz_size, ez_size=self.ez_size, gradient_estimator=self.gradient_estimator, pk_net_cls=self.pk_net_cls, pz_net_cls=self.pz_net_cls)
        ax = eps.ndim - 2
        S = nn.scan(cell, variable_broadcast="params", in_axes=ax, out_axes=ax,
                   split_rngs={"params": False})
        return S()(init, (xs, eps, mask))
        
class DSLDS(Module):
    latent_D: int = 10
    K: int = 10
    network_size: int = 64
    lstm_size: int = 64
    log_input: bool = False
    decoder_cls: ModuleDef = Decoder
    encoder_network_cls: Optional[ModuleDef] = None
    net_cls: ModuleDef = SmallELUNet
    pgm_hyperparameters: Dict = field(default_factory=dict)
    fz_size: int = 64
    ez_size: int = 64
    gradient_estimator: str = 'concrete'
    pk_net_cls: Any = None
    pz_net_cls: Any = None
    
    @compact
    def __call__(self, x, eval_mode=False, mask=None, masking_mode=2, n_iwae_samples=0, theta_rng=None, temp=0.1):
        with jax.default_matmul_precision('float32'):
            if self.log_input:
                x = jnp.log(x)

            z_rng = self.make_rng('sampler')
            z_rng, k_rng, mniw_rng, niw_rng, kappa_rng, alpha_rng = jax.random.split(z_rng, 6)

            # Instantiate the global SLDS parameters
            pgm = PGM_SLDS(self.latent_D, self.K, lambda x: x, name="pgm", **self.pgm_hyperparameters)
            _, prior_kl, global_natparams = pgm.expected_params()

            # Sample the global parameters from Q(\theta)
            niw_nat, mniw_nat, kappa_nat, alpha_nat = global_natparams
            Qp, Ab = jax.vmap(mniw.sample, in_axes=(0, 0, None))(mniw_nat, jax.random.split(mniw_rng, self.K), x.shape[0])
            Qp, Ab = jnp.swapaxes(Qp, 0, 1), jnp.swapaxes(Ab, 0, 1)
            start_Qp, start_mu = niw.sample(niw_nat, niw_rng, x.shape[0])
            Q, start_Q = inv_pd(Qp), inv_pd(start_Qp)
            kappa = dirichlet.sample(kappa_nat, kappa_rng, x.shape[0])
            alpha = dirichlet.sample(alpha_nat, alpha_rng, x.shape[0])
            pgm_params = (alpha, kappa, Ab, Q, start_mu, start_Q)
#             id_print(jax.tree_map(lambda x: jnp.isnan(x).any(), start_Q))
#             id_print(jax.tree_map(lambda x: jnp.isnan(x).any(), start_Qp))

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

            scanner = partial(DSLDSLatent, net_cls=self.net_cls, latent_D=self.latent_D, K=self.K, fz_size=self.fz_size, ez_size=self.ez_size, gradient_estimator=self.gradient_estimator, pk_net_cls=self.pk_net_cls, pz_net_cls=self.pz_net_cls,  name='pgm_inference')

            if n_iwae_samples > 0:
                eps_shape = xa.shape[:1] + (n_iwae_samples,) + xa.shape[1:-1] + (self.latent_D,)
                xa, mask = jnp.expand_dims(xa, 1), jnp.expand_dims(mask, 1)
            else:
                eps_shape = xa.shape[:-1] + (self.latent_D,)

            epsilon = tfd.Normal(jnp.zeros(eps_shape), jnp.ones(eps_shape)).sample(seed=z_rng)
            latent_seq = scanner()
            _, (z_sample, local_kl, iwae_kl, qk) = latent_seq((xa, epsilon, mask, temp, k_rng, pgm_params))

            #(z_sample, local_kl, iwae_kl, qk) = tree_map(lambda x: x.astype(jnp.float32), (z_sample, local_kl, iwae_kl, qk))

            local_kl_full = local_kl        
            likelihood = self.decoder_cls(x.shape[-1], name="decoder")(z_sample.astype(jnp.float32), eval_mode=eval_mode)
            local_kl = jnp.sum(local_kl) if n_iwae_samples == 0 else iwae_kl.sum(axis=-1)
            prior_kl = jnp.mean(prior_kl)

            if eval_mode:
                return likelihood, prior_kl, local_kl, (z_sample, local_kl_full, qk)
            else:
                return likelihood, prior_kl, local_kl, z_sample
        

class VQSLDSCell(Module):
    latent_D: int = 10
    K: int = 50
    beta: float = 0.25
    ez_cls: ModuleDef = NonlinearEZ
    net_cls: ModuleDef = SmallTanhNet
    fz_size: int = 64
    ez_size: int = 64
    
    @compact
    def __call__(self, carry, t_in):
        temp, rng, z_sample, k_sample, pgm_params = carry        
        (transition, start_pk, _, _, _, _) = pgm_params
                
        k_rng, z_rng, rng = jax.random.split(rng, 3)
        # Sample the next state using the VI dist.
        xt, eps, mask = t_in 
        
        z_sample_finite = jnp.where(jnp.isfinite(z_sample), z_sample, jnp.zeros_like(z_sample))
        k_sample_finite = jnp.where(jnp.isfinite(k_sample), k_sample, jnp.ones_like(k_sample))
        # Get the LDS params for the current state
        ez = self.net_cls(self.ez_size, self.latent_D)
        gt = ez(jnp.concatenate([z_sample_finite, xt], axis=-1))
        
        C = self.param('C', lambda rng: jax.random.normal(rng, (self.K, self.latent_D), dtype=jnp.float32))
        distances = jnp.sqrt(jnp.sum(jnp.abs(gt[:,jnp.newaxis,:] - C[jnp.newaxis,:,:]) ** 2, axis=-1))
        qk_sample_ind = jnp.argmin(distances, axis=-1)
        qk_sample = nn.one_hot(qk_sample_ind, self.K)
        qk = tfd.OneHotCategorical(probs=qk_sample)
                
        # Get the normalized transition matrix and the next state distribution
        pk_gammat = jnp.einsum('nk,nkj->nj', k_sample_finite, transition)
        pk_gammat = pk_gammat / pk_gammat.sum(axis=-1, keepdims=True)
        
        # Use a different state distribution for step 0
        pk_gammat = jnp.where(jnp.isfinite(k_sample), pk_gammat, start_pk)
        
        # Sample the next state using the prior
        pk = tfd.OneHotCategorical(probs=pk_gammat)
        pk_sample = pk.sample(seed=k_rng)
        k_sample = jnp.where(jnp.expand_dims(mask, axis=-1), qk_sample, pk_sample)
        
        # KL-divergence for the discrete states (pretend 
        discrete_iwae_kl = (qk.log_prob(k_sample) - pk.log_prob(k_sample))
        discrete_kl = tfd.kl_divergence(tfd.Categorical(probs=qk_sample), tfd.Categorical(probs=pk_gammat))
        
        z_sample = jnp.dot(k_sample, C)
        kl = jnp.sqrt(jnp.sum(jnp.abs(jax.lax.stop_gradient(gt) - z_sample) ** 2, axis=-1))
        kl = kl + self.beta * jnp.sqrt(jnp.sum(jnp.abs(gt - jax.lax.stop_gradient(z_sample)) ** 2, axis=-1))
        
        return (temp, rng, z_sample, k_sample, pgm_params), (z_sample, kl + discrete_kl, discrete_kl, qk_sample)
    
class VQSLDSLatent(Module):
    latent_D: int = 10
    K: int = 50
    beta: float = 0.25
    ez_cls: ModuleDef = NonlinearEZ
    net_cls: ModuleDef = SmallTanhNet
    fz_size: int = 64
    ez_size: int = 64
    
    @compact
    def __call__(self, inputs):
        xs, eps, mask, temp, rng, pgm_params = inputs
        init = temp, rng, jnp.nan * jnp.zeros(eps.shape[:-2] + (self.latent_D,)), jnp.nan * jnp.zeros(eps.shape[:-2] + (self.K,)), pgm_params 
        cell = partial(VQSLDSCell, ez_cls=self.ez_cls, latent_D=self.latent_D, K=self.K, beta=self.beta, fz_size=self.fz_size, ez_size=self.ez_size)
        ax = eps.ndim - 2
        S = nn.scan(cell, variable_broadcast="params", in_axes=ax, out_axes=ax,
                   split_rngs={"params": False})
        return S()(init, (xs, eps, mask))
    
class VQSLDS(Module):
    latent_D: int = 10
    K: int = 10
    network_size: int = 64
    lstm_size: int = 64
    decoder_cls: ModuleDef = Decoder
    encoder_network_cls: Optional[ModuleDef] = None
    ez_cls: ModuleDef = NonlinearEZ
    net_cls: ModuleDef = SmallTanhNet
    pgm_hyperparameters: Dict = field(default_factory=dict)
    fz_size: int = 64
    ez_size: int = 64
    beta: float = 0.25
    
    @compact
    def __call__(self, x, eval_mode=False, mask=None, masking_mode=2, n_iwae_samples=0, temp=0.1, theta_rng=None):
        z_rng = self.make_rng('sampler')
        z_rng, k_rng, mniw_rng, niw_rng, kappa_rng, alpha_rng = jax.random.split(z_rng, 6)

        # Instantiate the global SLDS parameters
        pgm = PGM_SLDS(self.latent_D, self.K, lambda x: x, name="pgm", **self.pgm_hyperparameters)
        _, prior_kl, global_natparams = pgm.expected_params()

        # Sample the global parameters from Q(\theta)
        niw_nat, mniw_nat, kappa_nat, alpha_nat = global_natparams
        Qp, Ab = jax.vmap(mniw.sample, in_axes=(0, 0, None))(mniw_nat, jax.random.split(mniw_rng, self.K), x.shape[0])
        Qp, Ab = jnp.swapaxes(Qp, 0, 1), jnp.swapaxes(Ab, 0, 1)
        start_Qp, start_mu = niw.sample(niw_nat, niw_rng, x.shape[0])
        Q, start_Q = inv_pd(Qp), inv_pd(start_Qp)
        kappa = dirichlet.sample(kappa_nat, kappa_rng, x.shape[0])
        alpha = dirichlet.sample(alpha_nat, alpha_rng, x.shape[0])
        pgm_params = (alpha, kappa, Ab, Q, start_mu, start_Q)


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

        scanner = partial(VQSLDSLatent, ez_cls=self.ez_cls, net_cls=self.net_cls, latent_D=self.latent_D, K=self.K, beta=self.beta, fz_size=self.fz_size, ez_size=self.ez_size, name='pgm_inference')

        if n_iwae_samples > 0:
            eps_shape = xa.shape[:1] + (n_iwae_samples,) + xa.shape[1:-1] + (self.latent_D,)
            xa, mask = jnp.expand_dims(xa, 1), jnp.expand_dims(mask, 1)
        else:
            eps_shape = xa.shape[:-1] + (self.latent_D,)

        epsilon = tfd.Normal(jnp.zeros(eps_shape), jnp.ones(eps_shape)).sample(seed=z_rng)
        latent_seq = scanner()
        _, (z_sample, local_kl, iwae_kl, qk) = latent_seq((xa, epsilon, mask, temp, k_rng, pgm_params))

        local_kl_full = local_kl        
        likelihood = self.decoder_cls(x.shape[-1], name="decoder")(z_sample, eval_mode=eval_mode)
        local_kl = jnp.sum(local_kl) if n_iwae_samples == 0 else iwae_kl.sum(axis=-1)
        prior_kl = jnp.mean(prior_kl)

        if eval_mode:
            return likelihood, prior_kl, local_kl, (z_sample, local_kl_full, qk)
        else:
            return likelihood, prior_kl, local_kl, z_sample