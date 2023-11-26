from jax import tree_map, vmap
import jax.numpy as jnp
from jax.random import split
from flax.linen import compact, initializers, softplus, Module
from distributions import mniw, niw, dirichlet
from utils import softminus, T, make_prior_fun, mask_potentials, straight_through_tuple, sample_and_logprob, inject_mingrads_pd
from typing import Callable, Any, Dict, Optional
ModuleDef = Any
from dataclasses import field
from inference.MP_Inference import sample_lds, lds_kl, lds_kl_full, lds_inference_homog, lds_transition_params_to_nat
from models.VAE import Encoder, SigmaDecoder
from functools import partial

class SIN_LDS_Gen(Module):
    latent_D: int
    nat_grads: bool = False
    S_0: float = 1.
    nu_0: float = 2.
    lam_0: float = 0.001
    M_0: float = 0.9
    S_init: float = 1.
    nu_init: float = 2.
    lam_init: float = 20.
    loc_init_sd: float = 0.2
    M_init: float = 0.9

    def setup(self):
        ### PRIORS

        # NIW for LDS initial state
        S_0v = jnp.identity(self.latent_D) * self.latent_D * self.S_0
        loc_0v = jnp.zeros((self.latent_D, 1))
        niw_0 = niw.moment_to_nat((S_0v, loc_0v, self.lam_0, self.latent_D + self.nu_0))
        if self.nat_grads:
            self.niw_prior_kl = make_prior_fun(niw_0, niw.logZ, straight_through_tuple(niw.expected_stats))
        else:
            self.niw_prior_kl = make_prior_fun(niw_0, niw.logZ, niw.expected_stats)
        self.niw_0 = niw_0

        # MNIW for LDS transitions
        V_0v = jnp.identity(self.latent_D + 1) * self.lam_0
        M_0v = jnp.eye(self.latent_D, self.latent_D + 1) * self.M_0
        mniw_0 = mniw.moment_to_nat((S_0v, M_0v, V_0v, self.latent_D + self.nu_0))
        if self.nat_grads:
            self.mniw_prior_kl = make_prior_fun(mniw_0, mniw.logZ, straight_through_tuple(mniw.expected_stats))
        else:
            self.mniw_prior_kl = make_prior_fun(mniw_0, mniw.logZ, mniw.expected_stats)
        self.mniw_0 = mniw_0

    def calc_prior_loss(self, params):
        niw_p, mniw_p = params
        return self.niw_prior_kl(niw_p) + self.mniw_prior_kl(mniw_p)

    @compact
    def __call__(self, x, n_iwae_samples=0, theta_rng=None):
        ### Initializations of prior params

        if self.nat_grads:
            # NIW
            S_p = jnp.identity(self.latent_D) * jnp.sqrt(self.latent_D * self.S_init)
            loc = jnp.zeros((self.latent_D, 1))
            lam_p = softminus(self.lam_init)
            nu_p = softminus(self.nu_init + 1.)

            S = jnp.matmul(S_p, T(S_p)) + jnp.identity(self.latent_D) * 1e-8
            lam = softplus(lam_p)
            nu = softplus(nu_p) + self.latent_D - 1
            niw_nat = self.param("niw", lambda rng: niw.moment_to_nat((S, loc, lam, nu)))

            # MNIW
            St_p = jnp.identity(self.latent_D) * jnp.sqrt(self.latent_D * self.S_init)
            St = jnp.matmul(St_p, T(St_p)) + jnp.identity(self.latent_D) * 1e-8
            V_p = jnp.identity(self.latent_D+1) * jnp.sqrt(self.lam_init)
            V = jnp.matmul(V_p, T(V_p)) + jnp.identity(self.latent_D + 1) * 1e-8
            nut_p = softminus(self.nu_init + 1.)
            nut = softplus(nut_p) + self.latent_D - 1

            def gen_mniw(key):
                off_diag = initializers.normal(stddev=self.loc_init_sd)(key, (self.latent_D, self.latent_D+1))
                diag_mask = jnp.eye(self.latent_D, self.latent_D + 1).astype(bool)
                M = jnp.where(diag_mask, self.M_0, off_diag)
                return mniw.moment_to_nat((St, M, V, nut))

            mniw_nat = self.param("mniw", gen_mniw)

            J, h, c, d = straight_through_tuple(niw.expected_stats)(niw_nat)
            E_mniw_params = straight_through_tuple(mniw.expected_stats)(mniw_nat)
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
            St_p = self.param("St", lambda rng: jnp.identity(self.latent_D) * jnp.sqrt(self.latent_D * self.S_init))
            def gen_M(key):
                off_diag = initializers.normal(stddev=self.loc_init_sd)(key, (self.latent_D, self.latent_D+1))
                diag_mask = jnp.eye(self.latent_D, self.latent_D + 1).astype(bool)
                return jnp.where(diag_mask, self.M_0, off_diag)
            M = self.param("M", gen_M)
            V_p = self.param("V", lambda rng: jnp.identity(self.latent_D+1) * jnp.sqrt(self.lam_init))
            nut_p = self.param("nut", lambda rng: softminus(self.nu_init + 1.))

            St = jnp.matmul(St_p, T(St_p)) + jnp.identity(self.latent_D) * 1e-8
            V = jnp.matmul(V_p, T(V_p)) + jnp.identity(self.latent_D + 1) * 1e-8
            nut = softplus(nut_p) + self.latent_D - 1
            mniw_nat = mniw.moment_to_nat((St, M, V, nut))

            J, h, c, d = niw.expected_stats(niw_nat)
            E_mniw_params = mniw.expected_stats(mniw_nat)

        if n_iwae_samples > 0:
            # sample from q(theta) and evaluate kl with the prior
            key, subkey = split(theta_rng)
            niw_sample, niw_global_kl = sample_and_logprob(self.niw_0, niw_nat, niw.logZ, 
                                                           partial(niw.sample_es, key=key), n=n_iwae_samples)
            mniw_sample, mniw_global_kl = sample_and_logprob(self.mniw_0, mniw_nat, mniw.logZ, 
                                                           partial(mniw.sample_es, key=subkey), n=n_iwae_samples)
            return (niw_sample, mniw_sample), niw_global_kl + mniw_global_kl

        prior_kl = self.calc_prior_loss((niw_nat, mniw_nat))
        ### Get expected potentials from PGM params.

        # NIW
        init = (J, h)
        E_init_normalizer = jnp.log(2 * jnp.pi)*self.latent_D/2 - c - d

        # MNIW
        # has mean M = [A|b] so we must break apart matrices into constituent parts
        transition_params = (jnp.expand_dims(E_mniw_params[2][-1,:-1],-1) * 2,
                     E_mniw_params[2][:-1,:-1],
                     E_mniw_params[1][:-1,:],
                     E_mniw_params[0],
                     jnp.expand_dims(E_mniw_params[1][-1,:],-1))
        E_trans_normalizer = x.shape[-2] * (jnp.log(2 * jnp.pi)*transition_params[2].shape[-1]/2 - (E_mniw_params[2][-1,-1] + E_mniw_params[-1]))
        E_prior_logZ = E_init_normalizer + E_trans_normalizer
        local_kl_fun = partial(lds_kl_full, prior_params=transition_params, prior_init=init, E_prior_logZ = E_prior_logZ)
        global_natparams = niw_nat, mniw_nat
        return local_kl_fun, prior_kl

class SIN_LDS(Module):
    latent_D: int
    S_init: float = 1.
    nu_init: float = 2.
    lam_init: float = 20.
    loc_init_sd: float = 0.2    
    M_init: float = 0.9
    nat_grads: bool = False
    S_0: float = 1.
    nu_0: float = 2.
    lam_0: float = 0.001
    M_0: float = 0.9


    @compact
    def expected_params(self, local_kl_fun):
        ### Initializations and converting from unconstrained space

        # NIW
        mu = self.param("loc", lambda rng: jnp.zeros((self.latent_D, 1)))
        tau_p = self.param("Tau", lambda rng: jnp.identity(self.latent_D))
        tau = jnp.matmul(tau_p, T(tau_p)) + jnp.identity(self.latent_D) * 1e-8
        tau_mu = jnp.matmul(tau, mu)
        J, h, c, d = (-tau/2, tau_mu, -jnp.matmul(T(mu), tau_mu)/2, jnp.linalg.slogdet(tau)[1]/2)
        init = (-2 * J, h)
        init_nat = (J, h)

        # MNIW
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
        
        pgm_potentials = init, transition_params
        if local_kl_fun is not None:
            local_kl_fun = partial(local_kl_fun, inference_params = transition_params_nat, inference_init=init_nat)
        return pgm_potentials, local_kl_fun

    def iwae(self, recog_potentials, rng, global_samples, n_iwae_samples):
        n = n_iwae_samples
        # sample from q(z)
        pgm_potentials, _ = self.expected_params(None)
        gaus_expected_stats, logZ, _ = lds_inference_homog(recog_potentials, *pgm_potentials)
        mapped_rng = split(rng,n_iwae_samples)
        zs = vmap(sample_lds, in_axes=[None, 0])(gaus_expected_stats, mapped_rng)

        def get_lds_logZ(niw_sample, mniw_sample):
            J, h, c, d = niw_sample
            init = (-2 * J, h)
            E_init_normalizer = jnp.log(2 * jnp.pi)*self.latent_D/2 - c - d

            transition_params = (jnp.expand_dims(mniw_sample[2][-1,:-1],-1) * -2,
                                 mniw_sample[2][:-1,:-1] * -2,
                                 mniw_sample[1][:-1,:],
                                 mniw_sample[0] * -2,
                                 jnp.expand_dims(mniw_sample[1][-1,:],-1))
            E_trans_normalizer = (recog_potentials[0].shape[0]-1) * (jnp.log(2 * jnp.pi)*transition_params[2].shape[-1]/2 - (mniw_sample[2][-1,-1] + mniw_sample[-1]))

            return (init, transition_params), E_init_normalizer + E_trans_normalizer

        niw_sample, mniw_sample = global_samples
        prior_params, prior_logZs = vmap(get_lds_logZ)(niw_sample, mniw_sample)

        # get difference between p(z|theta) and q(z)
        EXXT = vmap(vmap(lambda x: jnp.outer(x,x)))(zs)
        EX = jnp.expand_dims(zs, -1)
        EXXNT = vmap(vmap(lambda x,y: jnp.outer(x,y)))(zs[:,:-1], zs[:,1:])
        z_ss = (EXXT, EX, EXXNT)
        lds_kl_fun = vmap(lds_kl_full, in_axes=[None, 0, 0, 0, None, None, 0, None])
        lds_kl = lds_kl_fun(recog_potentials, z_ss, *lds_transition_params_to_nat(*prior_params),
                            *lds_transition_params_to_nat(*pgm_potentials), prior_logZs, logZ)

        return zs, lds_kl, (EXXT, EX, EXXNT)

    
    def __call__(self, recog_potentials, rng, local_kl_fun, n_iwae_samples = 0):
        if n_iwae_samples > 0:
            return self.iwae(recog_potentials, rng, local_kl_fun, n_iwae_samples)
        pgm_potentials, local_kl_fun = self.expected_params(local_kl_fun)
        gaus_expected_stats, logZ, _ = lds_inference_homog(recog_potentials, *pgm_potentials)
        local_kl = local_kl_fun(recog_potentials, gaus_expected_stats, logZ=logZ)
        z = sample_lds(gaus_expected_stats, rng)
        return z, local_kl, gaus_expected_stats

class SIN_LDS_VAE(Module):
    latent_D: int
    input_D: int
    log_input: bool = False
    encoder_cls: ModuleDef = Encoder
    decoder_cls: ModuleDef = SigmaDecoder
    #nat_grads: bool = False
    pgm_hyperparameters: Dict = field(default_factory=dict)

    @compact
    def __call__(self, x, eval_mode=False, mask=None, n_iwae_samples=0, theta_rng=None):
        if self.log_input:
            x = jnp.log(x)
        local_kl_fun, prior_kl = SIN_LDS_Gen(self.latent_D, name="pgm", **self.pgm_hyperparameters)(x, n_iwae_samples = n_iwae_samples, theta_rng = theta_rng)
        x_input = jnp.where(jnp.expand_dims(mask, -1), x, jnp.zeros_like(x)) if mask is not None else x
        recog_potentials = self.encoder_cls(self.latent_D, name="encoder")(x_input, eval_mode = eval_mode, mask=mask)
        if mask is not None:
            recog_potentials = mask_potentials(recog_potentials, mask)
            
        if n_iwae_samples > 0:
            if x.ndim == 3:
                rng = split(self.make_rng('sampler'),x.shape[0])
                inference_fun = partial(SIN_LDS(self.latent_D, name="vmp", **self.pgm_hyperparameters), local_kl_fun=local_kl_fun, n_iwae_samples=n_iwae_samples)
                z, local_kl, gaus_expected_stats = vmap(inference_fun)(recog_potentials, rng)
            else:
                rng = self.make_rng('sampler')
                z, local_kl, gaus_expected_stats = SIN_LDS(self.latent_D, name="vmp", **self.pgm_hyperparameters)(recog_potentials, rng, local_kl_fun, n_iwae_samples=n_iwae_samples)
            likelihood = self.decoder_cls(x.shape[-1], name="decoder")(z)
            # z will be B x N_iwae_samples x T x D; kl will be B x N_iwae_samples
            return likelihood, prior_kl, local_kl, z

        if x.ndim == 3:
            rng = split(self.make_rng('sampler'),x.shape[0])
            inference_fun = partial(SIN_LDS(self.latent_D, name="vmp", **self.pgm_hyperparameters), local_kl_fun=local_kl_fun)
            z, local_kl, gaus_expected_stats = vmap(inference_fun)(recog_potentials, rng)
            local_kl = local_kl.sum()
        else:
            rng = self.make_rng('sampler')
            z, local_kl, gaus_expected_stats = SIN_LDS(self.latent_D, name="vmp", **self.pgm_hyperparameters)(recog_potentials, rng, local_kl_fun)
        z, prior_kl, local_kl = tree_map(lambda x: x.astype(jnp.float32), (z, prior_kl, local_kl))
        likelihood = self.decoder_cls(x.shape[-1], name="decoder")(z)
        return likelihood, prior_kl, local_kl, gaus_expected_stats
