from jax import tree_map, vmap, value_and_grad
from jax.lax import stop_gradient
from jax.random import split
import jax.numpy as jnp
from flax.linen import compact, initializers, softplus, Module
from distributions import mniw, niw, dirichlet
from utils import softminus, T, make_prior_fun, mask_potentials, straight_through, straight_through_tuple
from typing import Callable, Any, Dict, Optional
ModuleDef = Any
from dataclasses import field
from inference.MP_Inference import sample_lds, hmm_inference, hmm_to_lds_mf, lds_transition_params_to_nat, lds_kl_full, hmm_kl_full, lds_inference, lds_inference_homog
import jax
from models.VAE import Encoder, SigmaDecoder
from functools import partial

class SIN_SLDS_Gen(Module):
    latent_D: int
    K: int
    nat_grads: bool = False
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
    time_homog: bool = True
    M_init: float = 0.9

    def setup(self):
        ### PRIORS

        # NIW for LDS initial state
        S_0v = jnp.identity(self.latent_D, dtype=jnp.float64) * self.latent_D * self.S_0
        loc_0v = jnp.zeros((self.latent_D, 1), dtype=jnp.float64)
        niw_0 = niw.moment_to_nat((S_0v, loc_0v, self.lam_0, self.latent_D + self.nu_0))
        if self.nat_grads:
            self.niw_prior_kl = make_prior_fun(niw_0, niw.logZ, straight_through_tuple(niw.expected_stats))
        else:
            self.niw_prior_kl = make_prior_fun(niw_0, niw.logZ, niw.expected_stats)

        # MNIW for LDS transitions
        V_0v = jnp.identity(self.latent_D + 1, dtype=jnp.float64) * self.lam_0
        M_0v = jnp.eye(self.latent_D, self.latent_D + 1, dtype=jnp.float64) * self.M_0
        mniw_0 = mniw.moment_to_nat((S_0v, M_0v, V_0v, self.latent_D + self.nu_0))
        if self.nat_grads:
            self.mniw_prior_kl = vmap(make_prior_fun(mniw_0, mniw.logZ, straight_through_tuple(mniw.expected_stats)))
        else:
            self.mniw_prior_kl = vmap(make_prior_fun(mniw_0, mniw.logZ, mniw.expected_stats))

        # Dirichlet for HMM
        kappa_0 = jnp.ones(self.K, dtype=jnp.float64) * self.kappa_0 - 1
        if self.nat_grads:
            self.kappa_prior_kl = make_prior_fun(kappa_0, dirichlet.logZ, straight_through(dirichlet.expected_stats))
        else:
            self.kappa_prior_kl = make_prior_fun(kappa_0, dirichlet.logZ, dirichlet.expected_stats)
        alpha_0 = jnp.ones((self.K,self.K), dtype=jnp.float64) * self.alpha_ij_0 + jnp.identity(self.K, dtype=jnp.float64) * (self.alpha_ii_0 - self.alpha_ij_0) - 1
        if self.nat_grads:
            self.alpha_prior_kl = make_prior_fun(alpha_0, lambda x: jnp.sum(dirichlet.logZ(x)),
                                                 straight_through(dirichlet.expected_stats))
        else:
            self.alpha_prior_kl = make_prior_fun(alpha_0, lambda x: jnp.sum(dirichlet.logZ(x)), dirichlet.expected_stats)

    def calc_prior_loss(self, params):
        niw_p, mniw_p, kappa_p, alpha_p = params
        return self.niw_prior_kl(niw_p) + jnp.sum(self.mniw_prior_kl(mniw_p)) + self.kappa_prior_kl(kappa_p) + jnp.sum(self.alpha_prior_kl(alpha_p))

    @compact
    def __call__(self):
        ### Initializations and converting from unconstrained space

        if self.nat_grads:
            # NIW
            S_p = jnp.identity(self.latent_D, dtype=jnp.float64) * jnp.sqrt(self.latent_D * self.S_init)
            loc = jnp.zeros((self.latent_D, 1), dtype=jnp.float64)
            lam_p = softminus(self.lam_init)
            nu_p = softminus(self.nu_init + 1.)

            S = jnp.matmul(S_p, T(S_p)) + jnp.identity(self.latent_D, dtype=jnp.float64) * 1e-8
            lam = softplus(lam_p)
            nu = softplus(nu_p) + self.latent_D - 1
            niw_nat = self.param("niw", lambda rng: niw.moment_to_nat((S, loc, lam, nu)))

            # MNIW
            St_p = jnp.tile(jnp.identity(self.latent_D, dtype=jnp.float64),(self.K,1,1)) * jnp.sqrt(self.latent_D * self.S_init)
            St = jnp.matmul(St_p, T(St_p)) + jnp.identity(self.latent_D, dtype=jnp.float64) * 1e-8
            V_p = jnp.tile(jnp.identity(self.latent_D+1, dtype=jnp.float64),(self.K,1,1)) * jnp.sqrt(self.lam_init)
            V = jnp.matmul(V_p, T(V_p)) + jnp.identity(self.latent_D + 1, dtype=jnp.float64) * 1e-8
            nut_p = softminus(jnp.ones(self.K, dtype=jnp.float64) + self.nu_init)
            nut = softplus(nut_p) + self.latent_D - 1

            def gen_mniw(key):
                off_diag = initializers.normal(stddev=self.loc_init_sd)(key, (self.K, self.latent_D, self.latent_D+1))
                diag_mask = jnp.tile(jnp.eye(self.latent_D, self.latent_D + 1, dtype=jnp.float64),(self.K,1,1)).astype(bool)
                M = jnp.where(diag_mask, self.M_0, off_diag)
                return vmap(mniw.moment_to_nat)((St, M, V, nut))

            mniw_nat = self.param("mniw", gen_mniw)

            # Dirichlet
            kappa_p = softminus(jnp.ones(self.K, dtype=jnp.float64) * self.kappa_init)
            kappa_nat = self.param("kappa", lambda rng: softplus(kappa_p) - 1)

            alpha_init = jnp.ones((self.K,self.K), dtype=jnp.float64) * self.alpha_ij_init + jnp.identity(self.K, dtype=jnp.float64) * (self.alpha_ii_init - self.alpha_ij_init)
            alpha_p = softminus(alpha_init)
            alpha_nat = self.param("alpha", lambda rng: softplus(alpha_p) - 1)

            # calculate expected statistics
            J, h, c, d = straight_through_tuple(niw.expected_stats)(niw_nat)
            E_mniw_params = vmap(straight_through_tuple(mniw.expected_stats))(mniw_nat)
            E_init_lps = jnp.expand_dims(straight_through(dirichlet.expected_stats)(kappa_nat),-1)
            E_trans_lps = straight_through(dirichlet.expected_stats)(alpha_nat)
        else:
            # NIW
            S_p = self.param("S", lambda rng: jnp.identity(self.latent_D) * jnp.sqrt(self.latent_D * self.S_init))
            loc = self.param("loc", lambda rng: jnp.zeros((self.latent_D, 1)))
            lam_p = self.param("lam", lambda rng: softminus(self.lam_init))
            nu_p = self.param("nu", lambda rng: softminus(self.nu_init + 1.))

            S = jnp.matmul(S_p, T(S_p)) + jnp.identity(self.latent_D) * 1e-8
            lam = softplus(lam_p)
            nu = softplus(nu_p) + self.latent_D - 1
            niw_nat = niw.moment_to_nat((S, loc, lam, nu))

            # MNIW
            St_p = self.param("St", lambda rng: jnp.tile(jnp.identity(self.latent_D),(self.K,1,1)) * jnp.sqrt(self.latent_D * self.S_init))
            def gen_M(key):
                off_diag = initializers.normal(stddev=self.loc_init_sd)(key, (self.K, self.latent_D, self.latent_D+1))
                diag_mask = jnp.tile(jnp.eye(self.latent_D, self.latent_D + 1),(self.K,1,1)).astype(bool)
                return jnp.where(diag_mask, self.M_0, off_diag)
            M = self.param("M", gen_M)
            V_p = self.param("V", lambda rng: jnp.tile(jnp.identity(self.latent_D+1),(self.K,1,1)) * jnp.sqrt(self.lam_init))
            nut_p = self.param("nut", lambda rng: softminus(jnp.ones(self.K) + self.nu_init))

            St = jnp.matmul(St_p, T(St_p)) + jnp.identity(self.latent_D) * 1e-8
            V = jnp.matmul(V_p, T(V_p)) + jnp.identity(self.latent_D + 1) * 1e-8
            nut = softplus(nut_p) + self.latent_D - 1
            mniw_nat = vmap(mniw.moment_to_nat)((St, M, V, nut))

            # Dirichlet
            kappa_p = self.param("kappa", lambda rng: softminus(jnp.ones(self.K) * self.kappa_init))
            alpha_init = jnp.ones((self.K,self.K)) * self.alpha_ij_init + jnp.identity(self.K) * (self.alpha_ii_init - self.alpha_ij_init)
            alpha_p = self.param("alpha", lambda rng: softminus(alpha_init))

            kappa_nat = softplus(kappa_p) - 1
            alpha_nat = softplus(alpha_p) - 1

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
        E_mniw_potentials = (jnp.expand_dims(E_mniw_params[2][:,-1,:-1],-1) * 2,
                             E_mniw_params[2][:,:-1,:-1],
                             E_mniw_params[1][:,:-1,:],
                             E_mniw_params[0],
                             jnp.expand_dims(E_mniw_params[1][:,-1,:],-1),
                             jnp.expand_dims(E_mniw_params[2][:,-1,-1] + E_mniw_params[-1], (-1,-2)))

        # Dirichlet
        pgm_potentials = E_mniw_potentials, init, E_init_normalizer, E_init_lps, E_trans_lps
        global_natparams = niw_nat, mniw_nat, kappa_nat, alpha_nat
        return pgm_potentials, self.calc_prior_loss(global_natparams)

class SIN_SLDS(Module):
    latent_D: int
    K: int
    nat_grads: bool = False
    S_0: float = 1.
    nu_0: float = 2.
    lam_0: float = 0.001
    M_0: float = 0.9
    kappa_0: float = 0.1
    alpha_ii_0: float = 100.
    alpha_ij_0: float = 0.5
    time_homog: bool = True
    S_init: float = 1.
    M_init: float = 0.9
    nu_init: float = 2.
    lam_init: float = 20.
    kappa_init: float = 1.
    alpha_ii_init: float = 1.
    alpha_ij_init: float = 0.9
    loc_init_sd: float = 0.2    

    @compact
    def expected_params(self, recog_potentials):
        N = recog_potentials[0].shape[0] - 1
        # NIW
        mu = self.param("loc", lambda rng: jnp.zeros((self.latent_D, 1)))
        tau_p = self.param("Tau", lambda rng: jnp.identity(self.latent_D))
        tau = jnp.matmul(tau_p, T(tau_p)) + jnp.identity(self.latent_D) * 1e-8
        tau_mu = jnp.matmul(tau, mu)
        J, h, c, d = (-tau/2, tau_mu, -jnp.matmul(T(mu), tau_mu)/2, jnp.linalg.slogdet(tau)[1]/2)
        init = (-2 * J, h)
        init_nat = (J, h)

        # MNIW
        if self.time_homog:
            lam_p = self.param("Lambda", lambda rng: jnp.identity(self.latent_D))
            lam = jnp.matmul(lam_p, T(lam_p)) + jnp.identity(self.latent_D) * 1e-8

            def gen_X(key):
                off_diag = initializers.normal(stddev=self.loc_init_sd)(key, (self.latent_D, self.latent_D+1))
                diag_mask = jnp.eye(self.latent_D, self.latent_D + 1).astype(bool)
                return jnp.where(diag_mask, self.M_init, off_diag)
            X = self.param("X", gen_X)

            XT_Lam = jnp.matmul(T(X),lam)
            E_mniw_params = (-lam/2, XT_Lam, -jnp.matmul(XT_Lam, X)/2, jnp.linalg.slogdet(lam)[1]/2)

            transition_params = (jnp.expand_dims(E_mniw_params[2][-1,:-1],-1) * -2,
                                 E_mniw_params[2][:-1,:-1] * -2,
                                 E_mniw_params[1][:-1,:],
                                 E_mniw_params[0] * -2,
                                 jnp.expand_dims(E_mniw_params[1][-1,:],-1))

            transition_params_nat = (jnp.expand_dims(E_mniw_params[2][-1,:-1],-1) * 2,
                         E_mniw_params[2][:-1,:-1],
                         E_mniw_params[1][:-1,:],
                         E_mniw_params[0],
                         jnp.expand_dims(E_mniw_params[1][-1,:],-1))
        else:
            lam_p = self.param("Lambda", lambda rng: jnp.tile(jnp.identity(self.latent_D), (N, 1, 1)))
            lam = jnp.matmul(lam_p, T(lam_p)) + jnp.identity(self.latent_D) * 1e-8

            def gen_X(key):
                off_diag = initializers.normal(stddev=self.loc_init_sd)(key, (N, self.latent_D, self.latent_D+1))
                diag_mask = jnp.eye(self.latent_D, self.latent_D + 1).astype(bool)
                return jnp.where(diag_mask, self.M_init, off_diag)
            X = self.param("X", gen_X)

            XT_Lam = jnp.matmul(T(X),lam)
            E_mniw_params = (-lam/2, XT_Lam, -jnp.matmul(XT_Lam, X)/2, jnp.linalg.slogdet(lam)[1]/2)

            transition_params = (jnp.expand_dims(E_mniw_params[2][:,-1,:-1],-1) * -2,
                                 E_mniw_params[2][:,:-1,:-1] * -2,
                                 E_mniw_params[1][:,:-1,:],
                                 E_mniw_params[0] * -2,
                                 jnp.expand_dims(E_mniw_params[1][:,-1,:],-1))

            transition_params_nat = (jnp.expand_dims(E_mniw_params[2][:,-1,:-1],-1) * 2,
                         E_mniw_params[2][:,:-1,:-1],
                         E_mniw_params[1][:,:-1,:],
                         E_mniw_params[0],
                         jnp.expand_dims(E_mniw_params[1][:,-1,:],-1))

        # Dirichlet
        kappa_p = softminus(jnp.ones(self.K) * self.kappa_init)
        alpha_init = jnp.ones((self.K,self.K)) * self.alpha_ij_init + jnp.identity(self.K) * (self.alpha_ii_init - self.alpha_ij_init)
        alpha_p = softminus(alpha_init)

        kappa_nat = softplus(kappa_p) - 1
        alpha_nat = softplus(alpha_p) - 1

        E_init_lps = self.param("init_lps", lambda rng: jnp.expand_dims(dirichlet.expected_stats(kappa_nat),-1))
        E_trans_lps = self.param("trans_lps", lambda rng: dirichlet.expected_stats(alpha_nat))
        
        lds_potentials = init, transition_params
        hmm_potentials = E_init_lps, E_trans_lps
        lds_kl_fun = partial(lds_kl_full, inference_params = transition_params_nat, inference_init=init_nat)
        return lds_potentials, hmm_potentials, lds_kl_fun

    def __call__(self, recog_potentials, rng, prior_potentials):
        E_mniw_params, prior_init, E_init_normalizer, E_init_lps_prior, E_trans_lps_prior = prior_potentials
        zeros = jnp.zeros((recog_potentials[0].shape[0] - 1, self.K))
        lds_potentials, hmm_potentials, lds_kl_fun = self.expected_params(recog_potentials)
        def hmm_mp(E_trans_lps):
            cat_es, logZ, _ = hmm_inference(hmm_potentials[0], E_trans_lps, zeros)
            return logZ, cat_es
        
        (hmm_logZ, cat_expected_stats), EZZNT = value_and_grad(hmm_mp, has_aux=True)(hmm_potentials[1])
        
        hmm_kl = hmm_kl_full(zeros, cat_expected_stats, hmm_logZ, E_init_lps_prior, E_trans_lps_prior, *hmm_potentials, EZZNT)
        
        if self.time_homog:
            gaus_expected_stats, gaus_logZ, _ = lds_inference_homog(recog_potentials, *lds_potentials)
        else:
            gaus_expected_stats, gaus_logZ, _ = lds_inference(recog_potentials, *lds_potentials)
        prior_gausparam, E_prior_logZ = hmm_to_lds_mf(cat_expected_stats, E_mniw_params, E_init_normalizer)
        prior_init_nat, prior_param_nat = lds_transition_params_to_nat(prior_init, prior_gausparam)
        lds_kl = lds_kl_fun(recog_potentials, gaus_expected_stats, prior_init_nat, prior_param_nat, E_prior_logZ = E_prior_logZ, logZ = gaus_logZ)
        z = sample_lds(gaus_expected_stats, rng)
        return z, (gaus_expected_stats, cat_expected_stats), lds_kl + hmm_kl

class SIN_SLDS_VAE(Module):
    latent_D: int
    input_D: int
    K: int
    log_input: bool = False
    encoder_cls: ModuleDef = Encoder
    decoder_cls: ModuleDef = SigmaDecoder
    time_homog: bool = True
    pgm_hyperparameters: Dict = field(default_factory=dict)

    @compact
    def __call__(self, x, eval_mode=False, mask=None):
        if self.log_input:
            x = jnp.log(x)
        prior_potentials, prior_kl = SIN_SLDS_Gen(self.latent_D, self.K, name="pgm", **self.pgm_hyperparameters)()
        x_input = jnp.where(jnp.expand_dims(mask, -1), x, jnp.zeros_like(x)) if mask is not None else x
        recog_potentials = self.encoder_cls(self.latent_D, name="encoder")(x_input, eval_mode = eval_mode, mask=mask)
        with jax.default_matmul_precision('float32'):
            if mask is not None:
                recog_potentials = mask_potentials(recog_potentials, mask)
            if x.ndim == 3:
                rng = split(self.make_rng('sampler'),x.shape[0])
                inference_fun = partial(SIN_SLDS(self.latent_D, self.K, time_homog=self.time_homog, name="vmp", **self.pgm_hyperparameters), prior_potentials = prior_potentials)
                z, cat_expected_stats, local_kl = vmap(inference_fun)(recog_potentials, rng)
                local_kl = local_kl.sum()
            else:
                rng = self.make_rng('sampler')
                z, cat_expected_stats, local_kl = SIN_SLDS(self.latent_D, self.K, time_homog=self.time_homog, name="vmp", **self.pgm_hyperparameters)(recog_potentials, rng, prior_potentials)
        z, prior_kl, local_kl = tree_map(lambda x: x.astype(jnp.float32), (z, prior_kl, local_kl))
        likelihood = self.decoder_cls(x.shape[-1], name="decoder")(z)
        return likelihood, prior_kl, local_kl, cat_expected_stats
