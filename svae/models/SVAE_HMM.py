from jax import tree_map, vmap, jit
from jax.lax import stop_gradient
from jax.random import split
import jax.numpy as jnp
from flax.linen import compact, initializers, softplus, Module
from distributions import mniw, niw, dirichlet, normal
from utils import softminus, make_prior_fun, mask_potentials, straight_through, straight_through_tuple, inject_mingrads_pd, inject_constgrads_pd, sample_and_logprob, sample_and_logprob_key, corr_param_inv
from typing import Callable, Any, Dict, Optional
ModuleDef = Any
from dataclasses import field
from functools import partial
from inference.HMM_MF_Inference import hmm_mf_kl, hmm_mf_kl_det, hmm_mf_kl_sur, hmm_mf_inference_itersolve, sample_hmm_mf
from inference.MP_Inference import cat_to_gaus_mf, gaus_to_cat_mf, hmm_inference, hmm_sample, hmm_kl_full, single_gaus_kl_det
from networks.encoders import Encoder
from networks.decoders import SigmaDecoder
from tensorflow_probability.substrates.jax import distributions as tfd
from jax.experimental.host_callback import id_print
import jax

class PGM_HMM(Module):
    latent_D: int
    K: int
    inference_fun: Callable
    nat_grads: bool = False
    new_nat_grads: bool = False
    drop_correction: bool = False
    S_0: float = 1.
    nu_0: float = 2.
    lam_0: float = 0.001
    kappa_0: float = 0.1
    alpha_ii_0: float = 100.
    alpha_ij_0: float = 0.5
    S_init: float = 1.
    nu_init: float = 2.
    lam_init: float = 20.
    kappa_init: float = 1.
    alpha_ii_init: float = 1.
    alpha_ij_init: float = 0.9
    loc_init_sd: float = 0.2 
    inf32: bool = True

    def setup(self):
        ### PRIORS

        # NIW for each Gaussian
        S_0v = jnp.identity(self.latent_D) * self.latent_D * self.S_0
        loc_0v = jnp.zeros((self.latent_D, 1))
        niw_0 = niw.moment_to_nat((S_0v, loc_0v, self.lam_0, self.latent_D + self.nu_0))
        if self.nat_grads or self.new_nat_grads:
            self.niw_prior_kl = vmap(make_prior_fun(niw_0, niw.logZ, straight_through_tuple(niw.expected_stats)))
        else:
            self.niw_prior_kl = vmap(make_prior_fun(niw_0, niw.logZ, niw.expected_stats))
        self.niw_0 = niw_0

        # Dirichlet for HMM
        kappa_0 = jnp.ones(self.K) * self.kappa_0 - 1
        if self.nat_grads or self.new_nat_grads:
            self.kappa_prior_kl = make_prior_fun(kappa_0, dirichlet.logZ, straight_through(dirichlet.expected_stats))
        else:
            self.kappa_prior_kl = make_prior_fun(kappa_0, dirichlet.logZ, dirichlet.expected_stats)
        self.kappa_p0 = kappa_0

        alpha_0 = jnp.ones((self.K,self.K)) * self.alpha_ij_0 + jnp.identity(self.K) * (self.alpha_ii_0 - self.alpha_ij_0) - 1
        if self.nat_grads or self.new_nat_grads:
            self.alpha_prior_kl = make_prior_fun(alpha_0, lambda x: jnp.sum(dirichlet.logZ(x)),
                                                 straight_through(dirichlet.expected_stats))
        else:
            self.alpha_prior_kl = make_prior_fun(alpha_0, lambda x: jnp.sum(dirichlet.logZ(x)), dirichlet.expected_stats)
        self.alpha_0 = alpha_0

    def calc_prior_loss(self, niw_p, kappa_p, alpha_p):
        return jnp.sum(self.niw_prior_kl(niw_p)) + self.kappa_prior_kl(kappa_p) + jnp.sum(self.alpha_prior_kl(alpha_p))

    @compact
    def expected_params(self):
        ### Initializations and converting from unconstrained space

        # NIW
        S = jnp.tile(corr_param_inv(jnp.identity(self.latent_D) * self.latent_D * self.S_init), (self.K,1,1))
        def gen_loc(key):
            return initializers.normal(stddev=self.loc_init_sd)(key, (self.K, self.latent_D, 1))
        lam = jnp.ones(self.K) * softminus(self.lam_init)
        nu = jnp.ones(self.K) * softminus(self.nu_init + 1.)

        # Dirichlet
        kappa = jnp.ones(self.K) * softminus(self.kappa_init)
        alpha = softminus(jnp.ones((self.K,self.K)) * self.alpha_ij_init + jnp.identity(self.K) * (self.alpha_ii_init - self.alpha_ij_init))

        if self.nat_grads:
            # Parameters in constrained space
            niw_nat = self.param("niw", lambda rng: vmap(niw.uton)((S, gen_loc(rng), lam, nu)))
            kappa_nat = self.param("kappa", lambda rng: dirichlet.uton(kappa))
            alpha_nat = self.param("alpha", lambda rng: dirichlet.uton(alpha))

            niw_nat = (vmap(inject_constgrads_pd)(niw_nat[0]), niw_nat[1], niw_nat[2], niw_nat[3])

            J, h, c, d = vmap(straight_through_tuple(niw.expected_stats))(niw_nat)
            E_init_lps = jnp.expand_dims(straight_through(dirichlet.expected_stats)(kappa_nat),-1)
            E_trans_lps = straight_through(dirichlet.expected_stats)(alpha_nat)
        elif self.new_nat_grads:
            # NIW
            S_p = self.param("S", lambda rng: S)
            loc_p = self.param("loc", lambda rng: gen_loc(rng))
            lam_p = self.param("lam", lambda rng: lam)
            nu_p = self.param("nu", lambda rng: nu)

            niw_nat = vmap(niw.uton_natgrad)((S_p, loc_p, lam_p, nu_p))
            niw_nat = (vmap(inject_mingrads_pd)(niw_nat[0]), niw_nat[1], niw_nat[2], niw_nat[3])

            # Dirichlet
            kappa_p = self.param("kappa", lambda rng: kappa)
            alpha_p = self.param("alpha", lambda rng: alpha)

            kappa_nat = dirichlet.uton_natgrad(kappa_p)
            alpha_nat = dirichlet.uton_natgrad(alpha_p)

            J, h, c, d = vmap(straight_through_tuple(niw.expected_stats))(niw_nat)
            E_init_lps = jnp.expand_dims(straight_through(dirichlet.expected_stats)(kappa_nat),-1)
            E_trans_lps = straight_through(dirichlet.expected_stats)(alpha_nat)

        else:
            # NIW
            S_p = self.param("S", lambda rng: S)
            loc_p = self.param("loc", lambda rng: gen_loc(rng))
            lam_p = self.param("lam", lambda rng: lam)
            nu_p = self.param("nu", lambda rng: nu)
            niw_nat = vmap(niw.uton)((S_p, loc_p, lam_p, nu_p))

            # Dirichlet
            kappa_p = self.param("kappa", lambda rng: kappa)
            alpha_p = self.param("alpha", lambda rng: alpha)

            kappa_nat = dirichlet.uton(kappa_p)
            alpha_nat = dirichlet.uton(alpha_p)

            J, h, c, d = vmap(niw.expected_stats)(niw_nat)
            E_init_lps = jnp.expand_dims(dirichlet.expected_stats(kappa_nat),-1)
            E_trans_lps = dirichlet.expected_stats(alpha_nat)

        gaus_params = (J, h)
        gaus_normalizer = jnp.log(2 * jnp.pi)*self.latent_D/2 - c - d

        pgm_potentials = gaus_params, gaus_normalizer, E_init_lps, E_trans_lps
        global_natparams = niw_nat, kappa_nat, alpha_nat
        return pgm_potentials, self.calc_prior_loss(*global_natparams), global_natparams

    def __call__(self, recog_potentials, key, initializer, n_forecast = 0, n_samples = 1):
        
        if n_forecast > 0:
            key, forecast_rng = split(key)

        # Get expectations of q(theta)
        pgm_potentials, prior_kl, global_natparams = self.expected_params()
        if self.inf32:
            recog_potentials, pgm_potentials = tree_map(lambda x: x.astype(jnp.float32), (recog_potentials, pgm_potentials))

        # PGM Inference
        if self.drop_correction:
            inference_params = tree_map(lambda x: stop_gradient(x), pgm_potentials)
        else:
            inference_params = pgm_potentials

        gaus_expected_stats, cat_expected_stats = self.inference_fun(recog_potentials, *inference_params, initializer)
        
        # Sample z
        if n_samples > 1:
            key = split(key,n_samples)
            z = vmap(sample_hmm_mf, in_axes=[None, 0])(gaus_expected_stats, rng)
        else:
            z = sample_hmm_mf(gaus_expected_stats, key)

        # calculate surrogate loss
        sur_loss = hmm_mf_kl_sur(recog_potentials, *inference_params, gaus_expected_stats, cat_expected_stats)

        # calculate local kl
        if self.drop_correction:
            local_kl = hmm_mf_kl_det(recog_potentials, pgm_potentials, inference_params,
                                  gaus_expected_stats, cat_expected_stats)
        else:
            local_kl = hmm_mf_kl(recog_potentials, *inference_params, gaus_expected_stats, cat_expected_stats)

        # forecast
        if n_forecast > 0:
            raise NotImplementedError

        return z, (gaus_expected_stats, cat_expected_stats), prior_kl, local_kl, sur_loss    
 
    def iwae(self, recog_potentials, key, initializer, theta_rng, n=1):
        pgm_potentials, _, global_natparams = self.expected_params()
        gaus_global, gaus_normalizer, E_init_lps, E_trans_lps = pgm_potentials

        # sample from q(theta) and evaluate kl from prior
        niw_key, kappa_key, alpha_key = split(theta_rng, 3)
        niw_key = split(niw_key, self.K)
        mapped_sampler = vmap(sample_and_logprob_key, in_axes=[None, 0, None, None, 0, None])
        niw_sample, niw_global_kl = mapped_sampler(self.niw_0, global_natparams[0], niw.logZ,
                                                   niw.sample_es, niw_key, n)
        niw_sample = tree_map(lambda x: x.swapaxes(0,1), niw_sample)
        niw_global_kl = niw_global_kl.sum(0)
        kappa_sample, kappa_global_kl = sample_and_logprob(self.kappa_p0, global_natparams[1], dirichlet.logZ,
                                                           partial(dirichlet.sample_es, key=kappa_key), n=n)
        alpha_logZ = lambda x: jnp.sum(dirichlet.logZ(x))
        alpha_sample, alpha_global_kl = sample_and_logprob(self.alpha_0, global_natparams[2], alpha_logZ,
                                                           partial(dirichlet.sample_es, key=kappa_key), n=n)

        # construct q(z)q(k), get logZ of each.
        gaus_expected_stats, cat_expected_stats = self.inference_fun(recog_potentials, *pgm_potentials, initializer)

        gaus_natparam, _ = cat_to_gaus_mf(cat_expected_stats, gaus_global, gaus_normalizer, recog_potentials)
        cat_natparam = gaus_to_cat_mf(gaus_expected_stats, gaus_global, gaus_normalizer)
        _, cat_logZ, (fmessages, _) = hmm_inference(E_init_lps, E_trans_lps, cat_natparam)

        # sample from q(k)
        k_key, z_key = split(key)
        k_key = split(k_key, n)
        ks = vmap(hmm_sample, in_axes=[None, None, 0, None])(cat_expected_stats, E_trans_lps, k_key, fmessages)

        # compute difference between q(k) and p(k|theta)
        Ekknt = vmap(vmap(jnp.outer))(ks[:,:-1], ks[:,1:])
        kl_fun = vmap(hmm_kl_full, in_axes = [None, 0, None, 0, 0, None, None, 0])
        hmm_kl = kl_fun(cat_natparam, ks, cat_logZ, jnp.expand_dims(kappa_sample, -1),
                        alpha_sample, E_init_lps, E_trans_lps, Ekknt)

        # sample from q(z)
        z_key = split(z_key, n)
        zs = vmap(sample_hmm_mf, in_axes=[None, 0])(gaus_expected_stats, z_key)

        # get params of p(z|theta,k) for our samples
        def get_prior_params(niw_sample, k):
            J, h, c, d = vmap(niw.expected_stats)(niw_sample)

            p_gaus_global = (J, h)
            p_gaus_normalizer = jnp.log(2 * jnp.pi)*self.latent_D/2 - c - d

            empty_rp = tree_map(lambda x: jnp.zeros_like(x), recog_potentials)
            return cat_to_gaus_mf(k, p_gaus_global, p_gaus_normalizer, empty_rp)

        prior_params = vmap(get_prior_params)(niw_sample, ks)

        # compute difference between q(z) and p(z|k,theta). logZ -> kl full
        EXXT = vmap(vmap(lambda x: jnp.outer(x,x)))(zs)
        EX = jnp.expand_dims(zs, -1)
        z_ss = (EXXT, EX)
        gaus_kl_fun = vmap(single_gaus_kl_det, in_axes=[0, None, 0, 0, None])
        gaus_kl = gaus_kl_fun(z_ss, gaus_natparam, *prior_params, recog_potentials)

        return zs, niw_global_kl + kappa_global_kl + alpha_global_kl, gaus_kl + hmm_kl

    def clip_params(self):
        raise NotImplementedError

class SVAE_HMM(Module):
    latent_D: int
    K: int
    input_D: int
    encoder_cls: ModuleDef = Encoder
    decoder_cls: ModuleDef = SigmaDecoder
    inference_fun: Callable = hmm_mf_inference_itersolve
    pgm_hyperparameters: Dict = field(default_factory=dict)
    log_input: bool = False
    autoreg: bool = False

    def setup(self):
        self.encoder = self.encoder_cls(self.latent_D, name="encoder")
        self.pgm = PGM_HMM(self.latent_D, self.K, self.inference_fun, name="pgm", **self.pgm_hyperparameters)
        self.decoder = self.decoder_cls(self.input_D, name="decoder")

    def __call__(self, x, eval_mode=False, mask=None, initializer = None, clip = False,
                 n_iwae_samples=0, theta_rng=None, n_forecast=0, n_samples=1, fixed_samples=None):

        if self.log_input:
            x = jnp.log(x)

        if not (mask is None):
            unscaled_mask = mask
            mask = jnp.where(mask > 0, jnp.ones_like(mask), jnp.zeros_like(mask))

        if clip:
            return self.pgm.clip_params() 

        x_input = jnp.where(jnp.expand_dims(mask, -1), x, jnp.zeros_like(x)) if mask is not None else x
        recog_potentials = self.encoder(x_input, eval_mode = eval_mode, mask=mask)

        if mask is not None:
            recog_potentials = mask_potentials(recog_potentials, mask)

        key = split(self.make_rng('sampler'),x.shape[0])
        if initializer is None:
            keys = vmap(split)(key)
            key, initializer = keys[:,0], keys[:,1]

        if n_iwae_samples > 0:
            with jax.default_matmul_precision('float32'):
                iwae_fun = vmap(self.pgm.iwae, in_axes=[0,0,0,None,None])
                z, prior_kl, local_kl = iwae_fun(recog_potentials, key, initializer, theta_rng, n_iwae_samples)
            likelihood = self.decoder(z.astype(jnp.float32), eval_mode=eval_mode)
            return likelihood, prior_kl, local_kl, z

        with jax.default_matmul_precision('float32'):
            pgm_fun = vmap(self.pgm, in_axes=[0,0,0,None,None])
            z, aux, prior_kl, local_kl, sur_loss = pgm_fun(recog_potentials, key, initializer, n_forecast, n_samples)
        prior_kl, local_kl, sur_loss = prior_kl.mean(), local_kl.sum(), sur_loss.sum()

        if self.autoreg:
            self_decoder_mask = None
            if fixed_samples is None:
                fixed_samples = jnp.zeros(z.shape)
            if not (mask is None):
                z = jnp.where(jnp.expand_dims(unscaled_mask, axis=-1) == 1, z, jnp.zeros_like(z))
                z = jnp.where(jnp.expand_dims(unscaled_mask, axis=-1) == 2, fixed_samples, z)
                self_decoder_mask = mask[...,:-1]#.at[...,1:].set(mask[...,:-1]).at[...,0].set(1)

            likelihood = self.decoder(x.astype(jnp.float32)[...,:-1,:], 
                                      z.astype(jnp.float32), eval_mode=eval_mode, mask = self_decoder_mask)
        else:
            likelihood = self.decoder(z.astype(jnp.float32), eval_mode=eval_mode)
        return likelihood, prior_kl, local_kl, (z, sur_loss) + aux