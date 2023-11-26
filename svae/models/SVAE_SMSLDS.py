from jax import tree_map, vmap, jit
from jax.lax import stop_gradient
from jax.random import split
import jax.numpy as jnp
from flax.linen import compact, initializers, softplus, Module
from distributions import mniw, niw, dirichlet, categorical
from utils import softminus, make_prior_fun, mask_potentials, straight_through, straight_through_tuple, inject_mingrads_pd, inject_constgrads_pd, sample_and_logprob, sample_and_logprob_key, corr_param_inv
from typing import Callable, Any, Dict, Optional
ModuleDef = Any
from dataclasses import field
from functools import partial
from inference.SLDS_Inference import sm_slds_inference_itersolve, sm_slds_kl, sm_slds_kl_sur, sample_slds_stable
from inference.MP_Inference import get_duration_lps
from networks.encoders import SigmaEncoder
from networks.decoders import SigmaDecoder
from tensorflow_probability.substrates.jax import distributions as tfd
from jax.experimental.host_callback import id_print
import jax
from jax.scipy.special import logsumexp
from networks.sequence import ReverseLSTM, LSTM
from distributions import normal

# if we don't think 32-bit is stable, we can remove mniw hard-coded rules
class PGM_SMSLDS(Module):
    latent_D: int
    K: int
    inference_fun: Callable
    T_max: int = 50
    nat_grads: bool = False
    new_nat_grads: bool = False
    drop_correction: bool = False
    S_0: float = 1.
    nu_0: float = 2.
    lam_0: float = 0.001
    M_0: float = 0.9
    kappa_0: float = 0.1
    alpha_ii_0: float = 100.
    alpha_ij_0: float = 0.5
    dur_0: float = 1.
    S_init: float = 1.
    nu_init: float = 2.
    lam_init: float = 20.
    kappa_init: float = 1.
    alpha_ii_init: float = 1.
    alpha_ij_init: float = 0.9
    dur_init: float = 1.
    loc_init_sd: float = 0.2
    r_max: int = 101
    no_bias: bool = False
    inf32: bool = True
    point_est: bool = False

    def setup(self):
        ### PRIORS

        # NIW for LDS initial state
        S_0v = jnp.identity(self.latent_D) * self.latent_D * self.S_0
        loc_0v = jnp.zeros((self.latent_D, 1))
        niw_0 = niw.moment_to_nat((S_0v, loc_0v, self.lam_0, self.latent_D + self.nu_0))
        if self.nat_grads or self.new_nat_grads:
            self.niw_prior_kl = make_prior_fun(niw_0, niw.logZ, straight_through_tuple(niw.expected_stats))
        else:
            self.niw_prior_kl = make_prior_fun(niw_0, niw.logZ, niw.expected_stats)
        self.niw_0 = niw_0

        # MNIW for LDS transitions
        V_0v = jnp.identity(self.latent_D + 1) * self.lam_0
        M_0v = jnp.eye(self.latent_D, self.latent_D + 1) * self.M_0
        mniw_0 = mniw.moment_to_nat((S_0v, M_0v, V_0v, self.latent_D + self.nu_0))
        if self.nat_grads or self.new_nat_grads:
            self.mniw_prior_kl = vmap(make_prior_fun(mniw_0, mniw.logZ, straight_through_tuple(mniw.expected_stats)))
        else:
            self.mniw_prior_kl = vmap(make_prior_fun(mniw_0, mniw.logZ, mniw.expected_stats))
        self.mniw_0 = mniw_0

        # Dirichlet for HMM
        kappa_0 = jnp.ones(self.K) * self.kappa_0 - 1
        if self.nat_grads or self.new_nat_grads:
            self.kappa_prior_kl = make_prior_fun(kappa_0, dirichlet.logZ, straight_through(dirichlet.expected_stats))
        else:
            self.kappa_prior_kl = make_prior_fun(kappa_0, dirichlet.logZ, dirichlet.expected_stats)
        self.kappa_p0 = kappa_0

        alpha_0 = jnp.ones((self.K,self.K - 1)) * self.alpha_ij_0 - 1
        if self.nat_grads or self.new_nat_grads:
            self.alpha_prior_kl = make_prior_fun(alpha_0, lambda x: jnp.sum(dirichlet.logZ(x)),
                                                 straight_through(dirichlet.expected_stats))
        else:
            self.alpha_prior_kl = make_prior_fun(alpha_0, lambda x: jnp.sum(dirichlet.logZ(x)), dirichlet.expected_stats)
        self.alpha_0 = alpha_0
        
        # Negative Binomial for Semi-Markov dynamics
        dur_alpha0 = jnp.ones((self.K, 2)) * self.dur_0
        if self.nat_grads or self.new_nat_grads:
            self.dur_prior_kl = make_prior_fun(dur_alpha0, lambda x: jnp.sum(dirichlet.logZ(x)),
                                               straight_through(dirichlet.expected_stats))
        else:
            self.dur_prior_kl = make_prior_fun(dur_alpha0, lambda x: jnp.sum(dirichlet.logZ(x)), dirichlet.expected_stats)
        self.dur_alpha0 = dur_alpha0

        dur_n0 = jnp.zeros((self.K, self.r_max))[:,::5]
        if self.nat_grads or self.new_nat_grads:
            self.dur_n_prior_kl = make_prior_fun(dur_n0, lambda x: jnp.sum(categorical.logZ(x)),
                                                 straight_through(categorical.expected_stats))
        else:
            self.dur_n_prior_kl = make_prior_fun(dur_n0, lambda x: jnp.sum(categorical.logZ(x)), 
                                                 categorical.expected_stats)
        self.dur_n0 = dur_n0

    def calc_prior_loss(self, params):
        niw_p, mniw_p, kappa_p, alpha_p, dur_alpha_p, dur_n_p = params
        return self.niw_prior_kl(niw_p) + jnp.sum(self.mniw_prior_kl(mniw_p)) + self.kappa_prior_kl(kappa_p) + self.alpha_prior_kl(alpha_p) + self.dur_prior_kl(dur_alpha_p) + self.dur_n_prior_kl(dur_n_p)

    @compact
    def expected_params(self):
        ### Initializations and converting from unconstrained space.

        # NIW
        S = corr_param_inv(jnp.identity(self.latent_D) * self.latent_D * self.S_init)
        loc = jnp.zeros((self.latent_D, 1))
        lam = jnp.ones(()) * softminus(self.lam_init)
        nu = jnp.ones(()) * softminus(self.nu_init + 1.)

        # MNIW
        St = jnp.tile(corr_param_inv(jnp.identity(self.latent_D) * self.latent_D * self.S_init), (self.K,1,1))
        def gen_M(key):
            off_diag = initializers.normal(stddev=self.loc_init_sd)(key, (self.K, self.latent_D, self.latent_D+1))
            diag_mask = jnp.tile(jnp.eye(self.latent_D, self.latent_D + 1),(self.K,1,1)).astype(bool)
            M = jnp.where(diag_mask, self.M_0, off_diag)
            return M
        V = jnp.tile(corr_param_inv(jnp.identity(self.latent_D + 1) * self.lam_init), (self.K,1,1))
        nut = softminus(self.nu_init + jnp.ones(self.K))

        # Dirichlet
        kappa = jnp.ones(self.K) * softminus(self.kappa_init)
        alpha = softminus(jnp.ones((self.K,self.K-1)) * self.alpha_ij_init)
        
        dur_alpha = jnp.ones((self.K, 2)) * softminus(self.dur_init)
        dur_n = jnp.zeros((self.K, self.r_max))[:,::5].at[:,0].set(5)

        if self.nat_grads:
            # Parameters in constrained space
            niw_nat = self.param("niw", lambda rng: niw.uton((S, loc, lam, nu)))
            mniw_nat = self.param("mniw", lambda rng: vmap(mniw.uton)((St, gen_M(rng), V, nut)))
            kappa_nat = self.param("kappa", lambda rng: dirichlet.uton(kappa))
            alpha_nat = self.param("alpha", lambda rng: dirichlet.uton(alpha))
            dur_alpha_nat = self.param("dur_alpha", lambda rng: dirichlet.uton(dur_alpha))
            dur_n_nat = self.param("dur_n", lambda rng: dur_n)

            niw_nat = (inject_mingrads_pd(niw_nat[0]), niw_nat[1], niw_nat[2], niw_nat[3])
            mniw_nat = (vmap(inject_mingrads_pd)(mniw_nat[0]), mniw_nat[1],
                        vmap(inject_mingrads_pd)(mniw_nat[2]), mniw_nat[3])

            # calculate expected statistics
            J, h, c, d = straight_through_tuple(niw.expected_stats)(niw_nat)
            E_mniw_params = vmap(straight_through_tuple(mniw.expected_stats))(mniw_nat)
            E_init_lps = jnp.expand_dims(straight_through(dirichlet.expected_stats)(kappa_nat),-1)
            E_trans_lps = straight_through(dirichlet.expected_stats)(alpha_nat)
            E_self_trans_lps = straight_through(dirichlet.expected_stats)(dur_alpha_nat)
            E_dur_n = straight_through(categorical.expected_stats)(dur_n_nat)

        elif self.new_nat_grads:
            # NIW
            S_p = self.param("S", lambda rng: S)
            loc_p = self.param("loc", lambda rng: loc)
            lam_p = self.param("lam", lambda rng: lam)
            nu_p = self.param("nu", lambda rng: nu)

            niw_nat = niw.uton_natgrad((S_p, loc_p, lam_p, nu_p))
            niw_nat = (inject_mingrads_pd(niw_nat[0]), niw_nat[1], niw_nat[2], niw_nat[3])

            # MNIW
            St_p = self.param("St", lambda rng: St)
            M_p = self.param("M", gen_M)
            V_p = self.param("V", lambda rng: V)
            nut_p = self.param("nut", lambda rng: nut)

            mniw_nat = vmap(mniw.uton_natgrad)((St_p, M_p, V_p, nut_p))
            mniw_nat = (vmap(inject_mingrads_pd)(mniw_nat[0]), mniw_nat[1], 
                        vmap(inject_mingrads_pd)(mniw_nat[2]), mniw_nat[3])

            # Dirichlet
            kappa_p = self.param("kappa", lambda rng: kappa)
            alpha_p = self.param("alpha", lambda rng: alpha)
            dur_alpha_p = self.param("dur_alpha", lambda rng: dur_alpha)
            dur_n_nat = self.param("dur_n", lambda rng: dur_n)

            kappa_nat = dirichlet.uton_natgrad(kappa_p)
            alpha_nat = dirichlet.uton_natgrad(alpha_p)
            dur_alpha_nat = dirichlet.uton_natgrad(dur_alpha_p)

            J, h, c, d = straight_through_tuple(niw.expected_stats)(niw_nat)
            E_mniw_params = vmap(straight_through_tuple(mniw.expected_stats))(mniw_nat)
            E_init_lps = jnp.expand_dims(straight_through(dirichlet.expected_stats)(kappa_nat),-1)
            E_trans_lps = straight_through(dirichlet.expected_stats)(alpha_nat)
            E_self_trans_lps = straight_through(dirichlet.expected_stats)(dur_alpha_nat)
            E_dur_n = straight_through(categorical.expected_stats)(dur_n_nat)

        elif self.point_est:
            raise NotImplementedError
        else:
            # NIW
            S_p = self.param("S", lambda rng: S)
            loc_p = self.param("loc", lambda rng: loc)
            lam_p = self.param("lam", lambda rng: lam)
            nu_p = self.param("nu", lambda rng: nu)
            niw_nat = niw.uton((S_p, loc, lam_p, nu_p))

            # MNIW
            St_p = self.param("St", lambda rng: St)
            M_p = self.param("M", gen_M)
            V_p = self.param("V", lambda rng: V)
            nut_p = self.param("nut", lambda rng: nut)
            mniw_nat = vmap(mniw.uton)((St_p, M_p, V_p, nut_p))

            # Dirichlet
            kappa_p = self.param("kappa", lambda rng: kappa)
            alpha_p = self.param("alpha", lambda rng: alpha)
            dur_alpha_p = self.param("dur_alpha", lambda rng: dur_alpha)
            dur_n_nat = self.param("dur_n", lambda rng: dur_n)

            kappa_nat = dirichlet.uton(kappa_p)
            alpha_nat = dirichlet.uton(alpha_p)
            dur_alpha_nat = dirichlet.uton(dur_alpha_p)

            J, h, c, d = niw.expected_stats(niw_nat)
            E_mniw_params = vmap(mniw.expected_stats)(mniw_nat)
            E_init_lps = jnp.expand_dims(dirichlet.expected_stats(kappa_nat),-1)
            E_trans_lps = dirichlet.expected_stats(alpha_nat)
            E_self_trans_lps = dirichlet.expected_stats(dur_alpha_nat)
            E_dur_n = categorical.expected_stats(dur_n_nat)

        ### Get expected potentials from PGM params.

        # NIW
        init = (-2 * J, h)
        E_init_normalizer = jnp.log(2 * jnp.pi)*self.latent_D/2 - c - d

        # MNIW
        # has mean M = [A|b] so we must break apart matrices into constituent parts
        bias_terms = (jnp.expand_dims(E_mniw_params[2][:,-1,:-1],-1) + jnp.expand_dims(E_mniw_params[2][:,:-1,-1],-1),
                      jnp.expand_dims(E_mniw_params[1][:,-1,:],-1),
                      jnp.expand_dims(E_mniw_params[2][:,-1,-1], (-1,-2)))
        bias_terms = [jnp.zeros_like(bt) for bt in bias_terms] if self.no_bias else bias_terms
        
        E_mniw_potentials = (bias_terms[0],
                             E_mniw_params[2][:,:-1,:-1],
                             E_mniw_params[1][:,:-1,:],
                             E_mniw_params[0],
                             bias_terms[1],
                             bias_terms[2] + jnp.expand_dims(E_mniw_params[-1], (-1,-2)))

        dur_potentials = vmap(get_duration_lps, in_axes=[0,0,None])(E_self_trans_lps, E_dur_n, self.T_max).T
        
        pgm_potentials = E_mniw_potentials, init, E_init_normalizer, E_init_lps, E_trans_lps, dur_potentials
        if self.point_est:
            return pgm_potentials, jnp.zeros(()), None
        global_natparams = niw_nat, mniw_nat, kappa_nat, alpha_nat, dur_alpha_nat, dur_n_nat
        return pgm_potentials, self.calc_prior_loss(global_natparams), global_natparams

    def __call__(self, recog_potentials, key, initializer, n_forecast = 0, n_samples=1):

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
            z, lds_logZ = vmap(sample_slds_stable, in_axes=[None, None, None, None, None, None, None, 0])(cat_expected_stats, recog_potentials, *inference_params[:-1], key)
            lds_logZ = lds_logZ[0]
        else:
            z, lds_logZ = sample_slds_stable(cat_expected_stats, recog_potentials, *inference_params[:-1], key)

        # calculate surrogate loss
        sur_loss = sm_slds_kl_sur(recog_potentials, *inference_params, gaus_expected_stats, cat_expected_stats, lds_logZ)

        # calculate local kl
        if self.drop_correction:
            raise NotImplementedError
        else:
            local_kl = sm_slds_kl(recog_potentials, *inference_params, gaus_expected_stats, cat_expected_stats, lds_logZ)

        # forecast
        if n_forecast > 0:
            raise NotImplementedError
#             forecasted_z = slds_forecast(z[-1], cat_expected_stats, global_natparams[1], 
#                                          global_natparams[-1], n_forecast, forecast_rng)
#             z = jnp.concatenate([z, forecasted_z], -2)

        return z, (gaus_expected_stats, cat_expected_stats), prior_kl, local_kl, sur_loss

    def iwae(self, recog_potentials, key, initializer, theta_rng, n=1):
        raise NotImplementedError

    def clip_params(self):
        raise NotImplementedError

class SVAE_SMSLDS(Module):
    latent_D: int
    K: int
    input_D: int = 96
    encoder_cls: ModuleDef = SigmaEncoder
    decoder_cls: ModuleDef = SigmaDecoder
    inference_fun: Callable = sm_slds_inference_itersolve
    pgm_hyperparameters: Dict = field(default_factory=dict)
    log_input: bool = False
    autoreg: bool = False

    def setup(self):
        self.encoder = self.encoder_cls(self.latent_D, name="encoder")
        self.pgm = PGM_SMSLDS(self.latent_D, self.K, self.inference_fun, name="pgm", **self.pgm_hyperparameters)
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
    