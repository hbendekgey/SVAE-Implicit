from jax import tree_map, vmap, jit
from jax.lax import stop_gradient
from jax.random import split
import jax.numpy as jnp
from flax.linen import compact, initializers, softplus, Module
from distributions import mniw, niw, dirichlet
from utils import softminus, make_prior_fun, mask_potentials, straight_through, straight_through_tuple, inject_mingrads_pd, inject_constgrads_pd, sample_and_logprob, sample_and_logprob_key, corr_param_inv
from typing import Callable, Any, Dict, Optional
ModuleDef = Any
from dataclasses import field
from functools import partial
from inference.SLDS_Inference import slds_inference_itersolve, slds_kl, slds_kl_det, slds_kl_sur, sample_slds_stable, slds_inference_itersolve_batched
from inference.MP_Inference import hmm_to_lds_mf, lds_to_hmm_mf, lds_inference, hmm_inference, hmm_kl_full, lds_kl_full, hmm_sample, slds_forecast, lds_transition_params_to_nat
from networks.encoders import Encoder
from networks.decoders import SigmaDecoder
from tensorflow_probability.substrates.jax import distributions as tfd
from jax.experimental.host_callback import id_print
import jax
from jax.scipy.special import logsumexp
from networks.sequence import ReverseLSTM, LSTM
from distributions import normal

# if we don't think 32-bit is stable, we can remove mniw hard-coded rules
class PGM_SLDS(Module):
    latent_D: int
    K: int
    inference_fun: Callable
    nat_grads: bool = False
    new_nat_grads: bool = True
    drop_correction: bool = False
    S_0: float = 1.
    nu_0: float = 2.
    lam_0: float = 0.001
    M_0: float = 0.9
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

        alpha_0 = jnp.ones((self.K,self.K)) * self.alpha_ij_0 + jnp.identity(self.K) * (self.alpha_ii_0 - self.alpha_ij_0) - 1
        if self.nat_grads or self.new_nat_grads:
            self.alpha_prior_kl = make_prior_fun(alpha_0, lambda x: jnp.sum(dirichlet.logZ(x)),
                                                 straight_through(dirichlet.expected_stats))
        else:
            self.alpha_prior_kl = make_prior_fun(alpha_0, lambda x: jnp.sum(dirichlet.logZ(x)), dirichlet.expected_stats)
        self.alpha_0 = alpha_0

    def calc_prior_loss(self, params):
        niw_p, mniw_p, kappa_p, alpha_p = params
        return self.niw_prior_kl(niw_p) + jnp.sum(self.mniw_prior_kl(mniw_p)) + self.kappa_prior_kl(kappa_p) + jnp.sum(self.alpha_prior_kl(alpha_p))

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
        alpha = softminus(jnp.ones((self.K,self.K)) * self.alpha_ij_init + jnp.identity(self.K) * (self.alpha_ii_init - self.alpha_ij_init))

        if self.nat_grads:
            # Parameters in constrained space
            niw_nat = self.param("niw", lambda rng: niw.uton((S, loc, lam, nu)))
            mniw_nat = self.param("mniw", lambda rng: vmap(mniw.uton)((St, gen_M(rng), V, nut)))
            kappa_nat = self.param("kappa", lambda rng: dirichlet.uton(kappa))
            alpha_nat = self.param("alpha", lambda rng: dirichlet.uton(alpha))

            niw_nat = (inject_mingrads_pd(niw_nat[0]), niw_nat[1], niw_nat[2], niw_nat[3])
            mniw_nat = (vmap(inject_mingrads_pd)(mniw_nat[0]), mniw_nat[1],
                        vmap(inject_mingrads_pd)(mniw_nat[2]), mniw_nat[3])

            # calculate expected statistics
            J, h, c, d = straight_through_tuple(niw.expected_stats)(niw_nat)
            E_mniw_params = vmap(straight_through_tuple(mniw.expected_stats))(mniw_nat)
            E_init_lps = jnp.expand_dims(straight_through(dirichlet.expected_stats)(kappa_nat),-1)
            E_trans_lps = straight_through(dirichlet.expected_stats)(alpha_nat)

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

            kappa_nat = dirichlet.uton_natgrad(kappa_p)
            alpha_nat = dirichlet.uton_natgrad(alpha_p)

            J, h, c, d = straight_through_tuple(niw.expected_stats)(niw_nat)
            E_mniw_params = vmap(straight_through_tuple(mniw.expected_stats))(mniw_nat)
            E_init_lps = jnp.expand_dims(straight_through(dirichlet.expected_stats)(kappa_nat),-1)
            E_trans_lps = straight_through(dirichlet.expected_stats)(alpha_nat)

        elif self.point_est:
            # niw
            mu = self.param("loc", lambda rng: jnp.zeros((self.latent_D, 1)))
            tau_p = self.param("Tau", lambda rng: jnp.identity(self.latent_D))
            tau = jnp.matmul(tau_p, tau_p.T) + jnp.identity(self.latent_D) * 1e-6
            tau_mu = jnp.matmul(tau, mu)
            J, h, c, d = (-tau/2, tau_mu, -jnp.matmul(mu.T, tau_mu).squeeze()/2, jnp.linalg.slogdet(tau)[1].squeeze()/2)

            # mniw
            lam_p = self.param("Lambda", lambda rng: jnp.tile(jnp.identity(self.latent_D), (self.K, 1, 1)))
            lam = jnp.matmul(lam_p, lam_p.swapaxes(-1,-2)) + jnp.identity(self.latent_D) * 1e-6
            X = self.param("X", lambda rng: jnp.tile(jnp.eye(self.latent_D, self.latent_D+1), (self.K, 1, 1)))

            def mniw_es(x, l):
                xtl = jnp.matmul(x.T,l)
                return (-l/2, xtl, -jnp.matmul(xtl, x)/2, jnp.linalg.slogdet(l)[1]/2)

            E_mniw_params = jax.vmap(mniw_es)(X,lam)

            pi0 = self.param("pi0", lambda rng: dirichlet.expected_stats(dirichlet.uton(kappa)))
            pi = self.param("pi", lambda rng: dirichlet.expected_stats(dirichlet.uton(alpha)))
            def normalize(ps):
                return ps - logsumexp(ps, -1, keepdims=True)
            E_init_lps = jnp.expand_dims(normalize(pi0), -1)
            E_trans_lps = normalize(pi)

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

            kappa_nat = dirichlet.uton(kappa_p)
            alpha_nat = dirichlet.uton(alpha_p)

            J, h, c, d = niw.expected_stats(niw_nat)
            E_mniw_params = vmap(mniw.expected_stats)(mniw_nat)
            E_init_lps = jnp.expand_dims(dirichlet.expected_stats(kappa_nat),-1)
            E_trans_lps = dirichlet.expected_stats(alpha_nat)

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

        # Dirichlet
        pgm_potentials = E_mniw_potentials, init, E_init_normalizer, E_init_lps, E_trans_lps
        if self.point_est:
            return pgm_potentials, jnp.zeros(()), None
        global_natparams = niw_nat, mniw_nat, kappa_nat, alpha_nat
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
            z, lds_logZ = vmap(sample_slds_stable, in_axes=[None, None, None, None, None, None, None, 0])(cat_expected_stats, recog_potentials, *inference_params, key)
            lds_logZ = lds_logZ[0]
        else:
            z, lds_logZ = sample_slds_stable(cat_expected_stats, recog_potentials, *inference_params, key)

        # calculate surrogate loss
        sur_loss = slds_kl_sur(recog_potentials, *inference_params, gaus_expected_stats, cat_expected_stats, lds_logZ)

        # calculate local kl
        if self.drop_correction:
            local_kl = slds_kl_det(recog_potentials, pgm_potentials, inference_params,
                                   gaus_expected_stats, cat_expected_stats, lds_logZ)
        else:
            local_kl = slds_kl(recog_potentials, *inference_params, gaus_expected_stats, cat_expected_stats, lds_logZ)

        # forecast
        if n_forecast > 0:
            forecasted_z = slds_forecast(z[-1], cat_expected_stats, global_natparams[1], 
                                         global_natparams[-1], n_forecast, forecast_rng)
            z = jnp.concatenate([z, forecasted_z], -2)

        return z, (gaus_expected_stats, cat_expected_stats), prior_kl, local_kl, sur_loss

    def iwae(self, recog_potentials, key, initializer, theta_rng, n=1):
        pgm_potentials, _, global_natparams = self.expected_params()
        E_mniw_params, init, E_init_normalizer, E_init_lps, E_trans_lps = pgm_potentials

        # sample from q(theta) and evaluate kl from prior
        niw_key, mniw_key, kappa_key, alpha_key = split(theta_rng, 4)
        mniw_key = split(mniw_key, self.K)
        niw_sample, niw_global_kl = sample_and_logprob(self.niw_0, global_natparams[0], niw.logZ,
                                               partial(niw.sample_es, key=niw_key), n=n)
        mapped_sampler = vmap(sample_and_logprob_key, in_axes=[None, 0, None, None, 0, None])
        mniw_sample, mniw_global_kl = mapped_sampler(self.mniw_0, global_natparams[1], mniw.logZ,
                                             mniw.sample_es, mniw_key, n)
        mniw_sample = tree_map(lambda x: x.swapaxes(0,1), mniw_sample)
        mniw_global_kl = mniw_global_kl.sum(0)
        kappa_sample, kappa_global_kl = sample_and_logprob(self.kappa_p0, global_natparams[2], dirichlet.logZ,
                                                           partial(dirichlet.sample_es, key=kappa_key), n=n)
        alpha_logZ = lambda x: jnp.sum(dirichlet.logZ(x))
        alpha_sample, alpha_global_kl = sample_and_logprob(self.alpha_0, global_natparams[3], alpha_logZ,
                                                           partial(dirichlet.sample_es, key=kappa_key), n=n)

        # construct q(z)q(k), get logZ of each.
        gaus_expected_stats, cat_expected_stats = self.inference_fun(recog_potentials, *pgm_potentials, initializer)

        gaus_natparam, _ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        cat_natparam = lds_to_hmm_mf(gaus_expected_stats, E_mniw_params)
        _, gaus_logZ, _ = lds_inference(recog_potentials, init, gaus_natparam)
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
        zs, _ = vmap(sample_slds_stable, in_axes=[None, None, None, None, None, None, None, 0])(cat_expected_stats, recog_potentials, *pgm_potentials, z_key)

        # get logZ of p(z|theta,k) for our samples
        def get_lds_logZ(niw_sample, mniw_sample, k):
            # NIW
            J, h, c, d = niw_sample
            init = (-2 * J, h)
            E_init_normalizer = jnp.log(2 * jnp.pi)*self.latent_D/2 - c - d

            # MNIW
            # has mean M = [A|b] so we must break apart matrices into constituent parts
            prior_mniws = (jnp.expand_dims(mniw_sample[2][:,-1,:-1],-1) + jnp.expand_dims(mniw_sample[2][:,:-1,-1],-1),
                                 mniw_sample[2][:,:-1,:-1],
                                 mniw_sample[1][:,:-1,:],
                                 mniw_sample[0],
                                 jnp.expand_dims(mniw_sample[1][:,-1,:],-1),
                                 jnp.expand_dims(mniw_sample[2][:,-1,-1] + mniw_sample[-1], (-1,-2)))

            transition_params, E_trans_normalizer = hmm_to_lds_mf(k, prior_mniws, E_init_normalizer)
            return (init, transition_params), E_init_normalizer + E_trans_normalizer

        prior_params, prior_logZs = vmap(get_lds_logZ)(niw_sample, mniw_sample, ks)


        # compute difference between q(z) and p(z|k,theta). logZ -> kl full
        EXXT = vmap(vmap(lambda x: jnp.outer(x,x)))(zs)
        EX = jnp.expand_dims(zs, -1)
        EXXNT = vmap(vmap(lambda x,y: jnp.outer(x,y)))(zs[:,:-1], zs[:,1:])
        z_ss = (EXXT, EX, EXXNT)
        lds_kl_fun = vmap(lds_kl_full, in_axes=[None, 0, 0, 0, None, None, 0, None])
        lds_kl = lds_kl_fun(recog_potentials, z_ss, *lds_transition_params_to_nat(*prior_params),
                            *lds_transition_params_to_nat(init, gaus_natparam), prior_logZs, gaus_logZ)

        return zs, niw_global_kl + mniw_global_kl + kappa_global_kl + alpha_global_kl, lds_kl + hmm_kl

    def clip_params(self):
        _, _, global_natparams = self.expected_params()
        priors = self.niw_0, self.mniw_0, self.kappa_p0, self.alpha_0

        # clip kappa and alpha params
        kappa_clipped, alpha_clipped = tree_map(lambda x,y: jnp.maximum(x,y), global_natparams[2:], priors[2:])
        kappa_ps = dirichlet.ntou(kappa_clipped)
        alpha_ps = dirichlet.ntou(alpha_clipped)

        # clip niw and mniw
        S, loc, lam, nu = niw.nat_to_moment(global_natparams[0])
        St, M, V, nut = vmap(mniw.nat_to_moment)(global_natparams[1])

        def clip_diag(p, val):
            return p.at[jnp.diag_indices_from(p)].set(jnp.maximum(jnp.diag(p), val))

        new_S = clip_diag(S, self.latent_D * self.S_0)
        new_lam = jnp.maximum(lam, self.lam_0)
        new_nu = jnp.maximum(nu, self.latent_D + self.nu_0)
        niw_ps = niw.ntou(niw.moment_to_nat((new_S, loc, new_lam, new_nu)))

        new_St = vmap(clip_diag, in_axes=[0,None])(St, self.latent_D * self.S_0)
        new_V = vmap(clip_diag, in_axes=[0,None])(V, self.lam_0)
        new_nut = jnp.maximum(nut, self.latent_D + self.nu_0)
        mniw_ps = vmap(mniw.ntou)(vmap(mniw.moment_to_nat)((new_St, M, new_V, new_nut)))

        return niw_ps, mniw_ps, kappa_ps, alpha_ps

class SVAE_SLDS(Module):
    latent_D: int
    K: int
    input_D: int = 96
    encoder_cls: ModuleDef = Encoder
    decoder_cls: ModuleDef = SigmaDecoder
    inference_fun: Callable = slds_inference_itersolve
    pgm_hyperparameters: Dict = field(default_factory=dict)
    log_input: bool = False
    autoreg: bool = False

    def setup(self):
        self.encoder = self.encoder_cls(self.latent_D, name="encoder")
        self.pgm = PGM_SLDS(self.latent_D, self.K, self.inference_fun, name="pgm", **self.pgm_hyperparameters)
        self.decoder = self.decoder_cls(self.input_D, name="decoder")

    def __call__(self, x, eval_mode=False, mask=None, initializer = None, clip = False,
                 n_iwae_samples=0, theta_rng=None, n_forecast=0, n_samples=1, fixed_samples=None):

        if not (mask is None):
            unscaled_mask = mask
            mask = jnp.where(mask > 0, jnp.ones_like(mask), jnp.zeros_like(mask))

        if clip:
            return self.pgm.clip_params() 

        if self.log_input:
            x = jnp.log(x)
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
    
class NetworkWrapper(Module):
    output_size: int
    network_cls: Optional[ModuleDef] = None
    
    @compact
    def __call__(self, *args, eval_mode=False):
        return self.network_cls(self.output_size, eval_mode=eval_mode)(*args)
    

# outdated, needs updating.
class TEACHER_FORCING_SVAE_SLDS(Module):
    latent_D: int
    K: int
    input_D: int = 96
    encoder_network_cls: Optional[ModuleDef] = None
    encoder_cls: ModuleDef = Encoder
    decoder_cls: ModuleDef = SigmaDecoder
    lstm_size: int = 64
    lstm_encoder: bool = True
    warmup_model: bool = False
    inference_fun: Callable = slds_inference_itersolve
    pgm_hyperparameters: Dict = field(default_factory=dict)
    
    def setup(self):
        self.encoder = self.encoder_cls(self.latent_D, name="encoder")
        self.encoder_network = NetworkWrapper(output_size=self.lstm_size, network_cls=self.encoder_network_cls, name="encoder_network")
        self.pgm = PGM_SLDS(self.latent_D, self.K, self.inference_fun, name="pgm", **self.pgm_hyperparameters)
        self.decoder = self.decoder_cls(self.input_D, name="decoder")
        self.ht_lstm = NetworkWrapper(output_size=self.lstm_size, network_cls=LSTM, name="ht")
        self.gt_lstm = NetworkWrapper(output_size=self.lstm_size, network_cls=ReverseLSTM, name="gt")

    def __call__(self, x, eval_mode=False, mask=None, initializer = None, clip = False,
                 n_iwae_samples=0, theta_rng=None, n_forecast = 0, n_samples=1, masking_mode=2):
        if clip:
            return self.pgm.clip_params() 
        
        # Deal with masking modes
        lengths = None
        if mask is None:
            mask = jnp.ones(x.shape[:2], dtype=jnp.int32)
        elif masking_mode == 0:
            lengths = mask.astype(jnp.int32).sum(axis=1)
            
        x = jnp.where(jnp.expand_dims(mask, -1), x, jnp.zeros_like(x))
        if masking_mode == 1:
            mask = jnp.ones_like(mask)
            
        xa = self.encoder_network(x, eval_mode=eval_mode)
        xa_shift = jnp.concatenate([jnp.zeros_like(xa[..., :1, :]), xa], axis=-2)[..., :-1, :]
        ht = self.ht_lstm(xa_shift, lengths, eval_mode=eval_mode)
        
        if self.lstm_encoder:
            gt = self.gt_lstm(jnp.concatenate([xa, ht], axis=-1), lengths, eval_mode=eval_mode)
            recog_potentials = self.encoder(gt, eval_mode=eval_mode)
        else:
            recog_potentials = self.encoder(xa, eval_mode=eval_mode)
        
        if self.warmup_model:
            mu, var = vmap(vmap(normal.nat_to_moment))(recog_potentials)
            q_z = tfd.Normal(mu.squeeze(-1), vmap(vmap(jnp.diag))(var))
            prior = tfd.Normal(jnp.zeros_like(q_z.loc), jnp.ones_like(q_z.loc))
            local_kl = tfd.kl_divergence(q_z, prior).sum()
            prior_kl = jnp.zeros_like(local_kl)
            local_kl_sur = jnp.zeros_like(local_kl)
            aux = tuple()

            z_rng = self.make_rng('sampler')
            z = q_z.sample(seed=z_rng)
            z, aux, prior_kl, local_kl, sur_loss = tree_map(lambda x: x.astype(jnp.float32), (z, aux, prior_kl, local_kl, local_kl_sur))
        else:
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
                    
                likelihood = self.decoder(jnp.concatenate([z, jnp.repeat(jnp.expand_dims(ht, 1), n_iwae_samples, axis=1)], axis=-1), eval_mode=eval_mode)
                return likelihood, prior_kl, local_kl, z

            with jax.default_matmul_precision('float32'):
                pgm_fun = vmap(self.pgm, in_axes=[0,0,0,None,None])
                z, aux, prior_kl, local_kl, sur_loss = pgm_fun(recog_potentials, key, initializer, n_forecast, n_samples)
            
            prior_kl, local_kl, sur_loss = prior_kl.mean(), local_kl.sum(), sur_loss.sum()
        
        if len(z.shape) > len(ht.shape):
            likelihood = self.decoder(jnp.concatenate([z, jnp.repeat(jnp.expand_dims(ht, 0), z.shape[0], axis=0)], axis=-1), eval_mode=eval_mode)
        else:
            likelihood = self.decoder(jnp.concatenate([z, ht], axis=-1), eval_mode=eval_mode)

        return likelihood, prior_kl, local_kl, (z, sur_loss) + aux

@jit
def clip_params(state):
    (S, loc, lam, nu), (St, M, V, nut), kappa, alpha = state.apply_fn({'params': state.params}, 0, clip=True)
    params = state.params['pgm']
    output = {}
    output['S'] = S
    output['loc'] = loc
    output['lam'] = lam
    output['nu'] = nu
    output['St'] = St
    output['M'] = M
    output['V'] = V
    output['nut'] = nut
    output['kappa'] = kappa
    output['alpha'] = alpha #or 1e4 * jnp.identity(10)
    return state.replace(params = state.params.copy({'pgm': output}))
