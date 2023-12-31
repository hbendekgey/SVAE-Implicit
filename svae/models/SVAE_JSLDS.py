from jax import tree_map, vmap, jit
from jax.lax import stop_gradient
from jax.random import split
import jax.numpy as jnp
from flax.linen import compact, initializers, softplus, Module
from distributions import mniw, niw, dirichlet
from utils import softminus, T, make_prior_fun, mask_potentials, straight_through, straight_through_tuple, inject_mingrads_pd, inject_constgrads_pd, sample_and_logprob, sample_and_logprob_key, corr_param_inv
from typing import Callable, Any, Dict, Optional
ModuleDef = Any
from jax.numpy import expand_dims, diag, zeros_like, ones_like

from dataclasses import field
from functools import partial
from inference.JumpingSLDS_Inference import jslds_inference_itersolve, jslds_kl_det, jslds_kl, jslds_kl_sur, sample_jslds_stable
from inference.MP_Inference import jumping_hmm_to_lds_mf, jumping_lds_to_hmm_mf, lds_inference, trans_hmm_inference, trans_hmm_sample, hmm_kl_full, lds_kl_full, lds_transition_params_to_nat
from networks.encoders import Encoder
from networks.decoders import SigmaDecoder
from tensorflow_probability.substrates.jax import distributions as tfd
from jax.experimental.host_callback import id_print
import jax

class PGM_JSLDS(Module):
    latent_D: int
    K: int
    inference_fun: Callable
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
    S_init: float = 1.
    nu_init: float = 2.
    lam_init: float = 20.
    kappa_init: float = 1.
    alpha_ii_init: float = 1.
    alpha_ij_init: float = 0.9
    loc_init_sd: float = 0.2
    no_bias: bool = False
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
        niw_p, mniw_p, mniw_jump_p, kappa_p, alpha_p = params
        return jnp.sum(self.niw_prior_kl(niw_p)) + jnp.sum(self.mniw_prior_kl(mniw_p)) + jnp.sum(self.mniw_prior_kl(mniw_jump_p)) + self.kappa_prior_kl(kappa_p) + jnp.sum(self.alpha_prior_kl(alpha_p))

    @compact
    def expected_params(self):
        ### Initializations and converting from unconstrained space
        
        # NIW
        S = jnp.tile(corr_param_inv(jnp.identity(self.latent_D) * self.latent_D * self.S_init), (self.K,1,1))
        def gen_loc(key):
            return initializers.normal(stddev=self.loc_init_sd)(key, (self.K, self.latent_D, 1))
        lam = jnp.ones(self.K) * softminus(self.lam_init)
        nu = jnp.ones(self.K) * softminus(self.nu_init + 1.)
        
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
            niw_nat = self.param("niw", lambda rng: vmap(niw.uton)((S, gen_loc(rng), lam, nu)))
            mniw_nat = self.param("mniw", lambda rng: vmap(mniw.uton)((St, gen_M(rng), V, nut)))
            mniw_jump_nat = self.param("mniw_jump", lambda rng: vmap(mniw.uton)((St, gen_M(rng), V, nut)))
            kappa_nat = self.param("kappa", lambda rng: dirichlet.uton(kappa))
            alpha_nat = self.param("alpha", lambda rng: dirichlet.uton(alpha))

            niw_nat = (vmap(inject_constgrads_pd)(niw_nat[0]), niw_nat[1], niw_nat[2], niw_nat[3])
            mniw_nat = (vmap(inject_constgrads_pd)(mniw_nat[0]), mniw_nat[1],
                        vmap(inject_constgrads_pd)(mniw_nat[2]), mniw_nat[3])
            mniw_jump_nat = (vmap(inject_constgrads_pd)(mniw_jump_nat[0]), mniw_jump_nat[1],
                             vmap(inject_constgrads_pd)(mniw_jump_nat[2]), mniw_jump_nat[3])

            J, h, c, d = vmap(straight_through_tuple(niw.expected_stats))(niw_nat)
            E_mniw_params = vmap(straight_through_tuple(mniw.expected_stats))(mniw_nat)
            E_mniw_jump_params = vmap(straight_through_tuple(mniw.expected_stats))(mniw_jump_nat)
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

            # MNIW
            St_p = self.param("St", lambda rng: St)
            M_p = self.param("M", gen_M)
            V_p = self.param("V", lambda rng: V)
            nut_p = self.param("nut", lambda rng: nut)

            mniw_nat = vmap(mniw.uton_natgrad)((St_p, M_p, V_p, nut_p))
            mniw_nat = (vmap(inject_mingrads_pd)(mniw_nat[0]), mniw_nat[1], 
                        vmap(inject_mingrads_pd)(mniw_nat[2]), mniw_nat[3])

            # MNIW jump
            St_pj = self.param("St_jump", lambda rng: St)
            M_pj = self.param("M_jump", gen_M)
            V_pj = self.param("V_jump", lambda rng: V)
            nut_pj = self.param("nut_jump", lambda rng: nut)

            mniw_jump_nat = vmap(mniw.uton_natgrad)((St_pj, M_pj, V_pj, nut_pj))
            mniw_jump_nat = (vmap(inject_mingrads_pd)(mniw_jump_nat[0]), mniw_jump_nat[1], 
                             vmap(inject_mingrads_pd)(mniw_jump_nat[2]), mniw_jump_nat[3])

            # Dirichlet
            kappa_p = self.param("kappa", lambda rng: kappa)
            alpha_p = self.param("alpha", lambda rng: alpha)

            kappa_nat = dirichlet.uton_natgrad(kappa_p)
            alpha_nat = dirichlet.uton_natgrad(alpha_p)

            J, h, c, d = vmap(straight_through_tuple(niw.expected_stats))(niw_nat)
            E_mniw_params = vmap(straight_through_tuple(mniw.expected_stats))(mniw_nat)
            E_mniw_jump_params = vmap(straight_through_tuple(mniw.expected_stats))(mniw_jump_nat)
            E_init_lps = jnp.expand_dims(straight_through(dirichlet.expected_stats)(kappa_nat),-1)
            E_trans_lps = straight_through(dirichlet.expected_stats)(alpha_nat)

        else:
            # NIW
            S_p = self.param("S", lambda rng: S)
            loc_p = self.param("loc", lambda rng: gen_loc(rng))
            lam_p = self.param("lam", lambda rng: lam)
            nu_p = self.param("nu", lambda rng: nu)
            niw_nat = vmap(niw.uton)((S_p, loc_p, lam_p, nu_p))

            # MNIW
            St_p = self.param("St", lambda rng: St)
            M_p = self.param("M", gen_M)
            V_p = self.param("V", lambda rng: V)
            nut_p = self.param("nut", lambda rng: nut)
            mniw_nat = vmap(mniw.uton)((St_p, M_p, V_p, nut_p))

            # MNIW jump
            St_pj = self.param("St_jump", lambda rng: St)
            M_pj = self.param("M_jump", gen_M)
            V_pj = self.param("V_jump", lambda rng: V)
            nut_pj = self.param("nut_jump", lambda rng: nut)
            mniw_jump_nat = vmap(mniw.uton)((St_pj, M_pj, V_pj, nut_pj))

            # Dirichlet
            kappa_p = self.param("kappa", lambda rng: kappa)
            alpha_p = self.param("alpha", lambda rng: alpha)

            kappa_nat = dirichlet.uton(kappa_p)
            alpha_nat = dirichlet.uton(alpha_p)

            J, h, c, d = vmap(niw.expected_stats)(niw_nat)
            E_mniw_params = vmap(mniw.expected_stats)(mniw_nat)
            E_mniw_jump_params = vmap(mniw.expected_stats)(mniw_jump_nat)
            E_init_lps = jnp.expand_dims(dirichlet.expected_stats(kappa_nat),-1)
            E_trans_lps = dirichlet.expected_stats(alpha_nat)

        ### Get expected potentials from PGM params.

        # NIW
        init = (J, h)
        E_init_normalizer = jnp.log(2 * jnp.pi)*self.latent_D/2 - c - d

        # MNIW
        # has mean M = [A|b] so we must break apart matrices into constituent parts
        bias_terms = (jnp.expand_dims(E_mniw_params[2][:,-1,:-1],-1) + jnp.expand_dims(E_mniw_params[2][:,:-1,-1],-1),
                      jnp.expand_dims(E_mniw_params[1][:,-1,:],-1),
                      jnp.expand_dims(E_mniw_params[2][:,-1,-1], (-1,-2)))
        bias_terms = [jnp.zeros_like(bt) for bt in bias_terms] if self.no_bias else bias_terms
        
        trans_params = (bias_terms[0],
                        E_mniw_params[2][:,:-1,:-1],
                        E_mniw_params[1][:,:-1,:],
                        E_mniw_params[0],
                        bias_terms[1],
                        bias_terms[2] + jnp.expand_dims(E_mniw_params[-1], (-1,-2)))

        # MNIW_jump
        bias_terms_j = (jnp.expand_dims(E_mniw_jump_params[2][:,-1,:-1],-1) + jnp.expand_dims(E_mniw_jump_params[2][:,:-1,-1],-1),
                      jnp.expand_dims(E_mniw_jump_params[1][:,-1,:],-1),
                      jnp.expand_dims(E_mniw_jump_params[2][:,-1,-1], (-1,-2)))
        bias_terms_j = [jnp.zeros_like(bt) for bt in bias_terms_j] if self.no_bias else bias_terms_j
        
        jump_params = (bias_terms_j[0],
                        E_mniw_jump_params[2][:,:-1,:-1],
                        E_mniw_jump_params[1][:,:-1,:],
                        E_mniw_jump_params[0],
                        bias_terms_j[1],
                        bias_terms_j[2] + jnp.expand_dims(E_mniw_jump_params[-1], (-1,-2)))

        pgm_potentials = (init, E_init_normalizer, jump_params, trans_params), (E_init_lps, E_trans_lps)
        global_natparams = niw_nat, mniw_nat, mniw_jump_nat, kappa_nat, alpha_nat
        return pgm_potentials, self.calc_prior_loss(global_natparams), global_natparams

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
            z, lds_logZ = vmap(sample_jslds_stable, in_axes=[None, None, None, None, None, None, None, 0])(cat_expected_stats, recog_potentials, *inference_params, key)
            lds_logZ = lds_logZ[0]
        else:
            z, lds_logZ = sample_jslds_stable(cat_expected_stats, recog_potentials, *inference_params, key)

        # calculate surrogate loss
        sur_loss = jslds_kl_sur(recog_potentials, *inference_params, gaus_expected_stats, cat_expected_stats, lds_logZ)

        # calculate local kl
        if self.drop_correction:
            local_kl = jslds_kl_det(recog_potentials, pgm_potentials, inference_params,
                                   gaus_expected_stats, cat_expected_stats, lds_logZ)
        else:
            local_kl = jslds_kl(recog_potentials, *inference_params, gaus_expected_stats, cat_expected_stats, lds_logZ)

        # forecast
        if n_forecast > 0:
            raise NotImplementedError

        return z, (gaus_expected_stats, cat_expected_stats[1].sum(-2)), prior_kl, local_kl, sur_loss

    def iwae(self, recog_potentials, key, initializer, theta_rng, n=1):
        pgm_potentials, _, global_natparams = self.expected_params()
        lds_params, hmm_params = pgm_potentials

        # sample from q(theta) and evaluate kl from prior
        niw_key, mniw_key, kappa_key, alpha_key = split(theta_rng, 4)
        niw_key, mniw_key = split(niw_key, self.K), split(mniw_key, self.K)
        mapped_sampler = vmap(sample_and_logprob_key, in_axes=[None, 0, None, None, 0, None])
        niw_sample, niw_global_kl = mapped_sampler(self.niw_0, global_natparams[0], niw.logZ,
                                                   niw.sample_es, niw_key, n)
        mniw_sample, mniw_global_kl = mapped_sampler(self.mniw_0, global_natparams[1], mniw.logZ,
                                             mniw.sample_es, mniw_key, n)
        niw_sample = tree_map(lambda x: x.swapaxes(0,1), niw_sample)
        mniw_sample = tree_map(lambda x: x.swapaxes(0,1), mniw_sample)
        niw_global_kl = niw_global_kl.sum(0)
        mniw_global_kl = mniw_global_kl.sum(0)
        kappa_sample, kappa_global_kl = sample_and_logprob(self.kappa_p0, global_natparams[2], dirichlet.logZ,
                                                           partial(dirichlet.sample_es, key=kappa_key), n=n)
        alpha_logZ = lambda x: jnp.sum(dirichlet.logZ(x))
        alpha_sample, alpha_global_kl = sample_and_logprob(self.alpha_0, global_natparams[3], alpha_logZ,
                                                           partial(dirichlet.sample_es, key=kappa_key), n=n)

        # construct q(z)q(k), get logZ of each.
        gaus_expected_stats, cat_expected_stats = self.inference_fun(recog_potentials, *pgm_potentials, initializer)

        gaus_natparam, _ = jumping_hmm_to_lds_mf(cat_expected_stats, *lds_params)
        cat_natparam = jumping_lds_to_hmm_mf(gaus_expected_stats, *lds_params)
        _, gaus_logZ, _ = lds_inference(recog_potentials, *gaus_natparam)
        _, hmm_logZ = trans_hmm_inference(*tree_map(lambda x,y: x+y, cat_natparam, hmm_params))

        # sample from q(k)
        k_key, z_key = split(key)
        k_key = split(k_key, n)
        ks, k_margs = vmap(trans_hmm_sample, in_axes=[None, None, 0])(*cat_expected_stats, k_key)

        # compute difference between q(k) and p(k|theta)
        kl_fun = vmap(hmm_kl_full, in_axes = [0, 0, None, 0, 0, None, None, 0])
        hmm_kl = kl_fun(jnp.zeros_like(k_margs), k_margs, hmm_logZ, jnp.expand_dims(kappa_sample, -1),
                        alpha_sample, *cat_natparam, ks[1])

        # sample from q(z)
        z_key = split(z_key, n)
        zs, _ = vmap(sample_jslds_stable, in_axes=[None, None, None, None, 0])(cat_expected_stats, recog_potentials, *pgm_potentials, z_key)

        # get logZ of p(z|theta,k) for our samples TODO FIX
        def get_lds_logZ(niw_sample, mniw_sample, k):
            # NIW
            J, h, c, d = niw_sample
            prior_niws = (J, h)
            prior_normalizers = jnp.log(2 * jnp.pi)*self.latent_D/2 - c - d

            # MNIW
            # has mean M = [A|b] so we must break apart matrices into constituent parts
            prior_mniws = (jnp.expand_dims(mniw_sample[2][:,-1,:-1],-1) + jnp.expand_dims(mniw_sample[2][:,:-1,-1],-1),
                                 mniw_sample[2][:,:-1,:-1],
                                 mniw_sample[1][:,:-1,:],
                                 mniw_sample[0],
                                 jnp.expand_dims(mniw_sample[1][:,-1,:],-1),
                                 jnp.expand_dims(mniw_sample[2][:,-1,-1] + mniw_sample[-1], (-1,-2)))

            return jumping_hmm_to_lds_mf(k, prior_niws, prior_normalizers, prior_mniws)

        prior_params, prior_logZs = vmap(get_lds_logZ)(niw_sample, mniw_sample, ks)

        # compute difference between q(z) and p(z|k,theta). logZ -> kl full
        EXXT = vmap(vmap(lambda x: jnp.outer(x,x)))(zs)
        EX = jnp.expand_dims(zs, -1)
        EXXNT = vmap(vmap(lambda x,y: jnp.outer(x,y)))(zs[:,:-1], zs[:,1:])
        z_ss = (EXXT, EX, EXXNT)
        lds_kl_fun = vmap(lds_kl_full, in_axes=[None, 0, 0, 0, None, None, 0, None])
        lds_kl = lds_kl_fun(recog_potentials, z_ss, *lds_transition_params_to_nat(*prior_params),
                            *lds_transition_params_to_nat(*gaus_natparam), prior_logZs, gaus_logZ)

        return zs, niw_global_kl + mniw_global_kl + kappa_global_kl + alpha_global_kl, lds_kl + hmm_kl

class SVAE_JSLDS(Module):
    latent_D: int
    K: int
    input_D: int
    encoder_cls: ModuleDef = Encoder
    decoder_cls: ModuleDef = SigmaDecoder
    inference_fun: Callable = jslds_inference_itersolve
    pgm_hyperparameters: Dict = field(default_factory=dict)
    log_input: bool = False

    def setup(self):
        self.encoder = self.encoder_cls(self.latent_D, name="encoder")
        self.pgm = PGM_JSLDS(self.latent_D, self.K, self.inference_fun, name="pgm", **self.pgm_hyperparameters)
        self.decoder = self.decoder_cls(self.input_D, name="decoder")

    @compact
    def __call__(self, x, eval_mode=False, mask=None, initializer = None
                 n_iwae_samples=0, theta_rng=None, n_forecast=0, n_samples=1):
        
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

        likelihood = self.decoder(z.astype(jnp.float32), eval_mode=eval_mode)
        return likelihood, prior_kl, local_kl, (z, sur_loss) + aux