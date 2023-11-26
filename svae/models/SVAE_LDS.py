from jax import tree_map, vmap
from jax.lax import stop_gradient
import jax.numpy as jnp
from jax.random import split
from flax.linen import compact, initializers, softplus, Module
from distributions import mniw, niw, dirichlet
from utils import softminus, T, make_prior_fun, mask_potentials, straight_through_tuple, sample_and_logprob, inject_mingrads_pd, inject_constgrads_pd, corr_param_inv
from functools import partial
from typing import Callable, Any, Dict
ModuleDef = Any
from dataclasses import field
from inference.MP_Inference import sample_lds, lds_inference_homog, lds_kl, lds_kl_full, lds_transition_params_to_nat, lds_forecast, lds_kl_surr
from networks.encoders import Encoder
from networks.decoders import SigmaDecoder
from distributions.tfp import NormalInverseWishart, MatrixNormalInverseWishart
import jax

class PGM_LDS(Module):
    latent_D: int
    nat_grads: bool = False
    drop_correction: bool = False
    new_nat_grads: bool = True
    S_0: float = 1.
    nu_0: float = 2.
    lam_0: float = 0.001
    M_0: float = 0.9
    S_init: float = 1.
    nu_init: float = 2.
    lam_init: float = 20.
    loc_init_sd: float = 0.2
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
            self.mniw_prior_kl = make_prior_fun(mniw_0, mniw.logZ, straight_through_tuple(mniw.expected_stats))
        else:
            self.mniw_prior_kl = make_prior_fun(mniw_0, mniw.logZ, mniw.expected_stats)
        self.mniw_0 = mniw_0

    def calc_prior_loss(self, params):
        niw_p, mniw_p = params
        return self.niw_prior_kl(niw_p) + self.mniw_prior_kl(mniw_p)

    @compact
    def expected_params(self, x):
        ### Initializations and converting from unconstrained space

        # NIW
        S = corr_param_inv(jnp.identity(self.latent_D) * self.latent_D * self.S_init)
        loc = jnp.zeros((self.latent_D, 1))
        lam = jnp.ones(()) * softminus(self.lam_init)
        nu = jnp.ones(()) * softminus(self.nu_init + 1.)

        # MNIW
        St = corr_param_inv(jnp.identity(self.latent_D) * self.latent_D * self.S_init)
        def gen_M(key):
            off_diag = initializers.normal(stddev=self.loc_init_sd)(key, (self.latent_D, self.latent_D+1))
            diag_mask = jnp.eye(self.latent_D, self.latent_D + 1).astype(bool)
            return jnp.where(diag_mask, self.M_0, off_diag)
        V = corr_param_inv(jnp.identity(self.latent_D + 1) * self.lam_init)
        nut = jnp.ones(()) * softminus(self.nu_init + 1.)

        if self.nat_grads:
            # Parameters in constrained space
            niw_nat = self.param("niw", lambda rng: niw.uton((S, loc, lam, nu)))
            mniw_nat = self.param("mniw", lambda rng: mniw.uton((St, gen_M(rng), V, nut)))

            niw_nat  = (inject_mingrads_pd(niw_nat[0]), niw_nat[1], niw_nat[2], niw_nat[3])
            mniw_nat = (inject_mingrads_pd(mniw_nat[0]), mniw_nat[1],
                        inject_mingrads_pd(mniw_nat[2]), mniw_nat[3])

            J, h, c, d = straight_through_tuple(niw.expected_stats)(niw_nat)
            E_mniw_params = straight_through_tuple(mniw.expected_stats)(mniw_nat)
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

            mniw_nat = mniw.uton_natgrad((St_p, M_p, V_p, nut_p))
            mniw_nat = (inject_mingrads_pd(mniw_nat[0]), mniw_nat[1], 
                        inject_mingrads_pd(mniw_nat[2]), mniw_nat[3])

            J, h, c, d = straight_through_tuple(niw.expected_stats)(niw_nat)
            E_mniw_params = straight_through_tuple(mniw.expected_stats)(mniw_nat)
        elif self.point_est:
            # niw
            mu = self.param("loc", lambda rng: jnp.zeros((self.latent_D, 1)))
            tau_p = self.param("Tau", lambda rng: jnp.identity(self.latent_D) * jnp.sqrt(self.nu_init/self.S_init))
            tau = jnp.matmul(tau_p, tau_p.T) + jnp.identity(self.latent_D) * 1e-6
            tau_mu = jnp.matmul(tau, mu)
            J, h, c, d = (-tau/2, tau_mu, -jnp.matmul(mu.T, tau_mu).squeeze()/2, jnp.linalg.slogdet(tau)[1].squeeze()/2)

            # mniw
            lam_p = self.param("Lambda", lambda rng: jnp.identity(self.latent_D) * jnp.sqrt(self.nu_init/self.S_init))
            lam = jnp.matmul(lam_p, lam_p.T) + jnp.identity(self.latent_D) * 1e-6
            X = self.param("X", lambda rng: jnp.eye(self.latent_D, self.latent_D+1) * self.M_0)

            def mniw_es(x, l):
                xtl = jnp.matmul(x.T,l)
                return (-l/2, xtl, -jnp.matmul(xtl, x)/2, jnp.linalg.slogdet(l)[1]/2)

            E_mniw_params = mniw_es(X,lam)
        else:
            # NIW
            S_p = self.param("S", lambda rng: S)
            loc_p = self.param("loc", lambda rng: loc)
            lam_p = self.param("lam", lambda rng: lam)
            nu_p = self.param("nu", lambda rng: nu)

            niw_nat = niw.uton((S_p, loc_p, lam_p, nu_p))

            # MNIW
            St_p = self.param("St", lambda rng: St)
            M_p = self.param("M", gen_M)
            V_p = self.param("V", lambda rng: V)
            nut_p = self.param("nut", lambda rng: nut)

            mniw_nat = mniw.uton((St_p, M_p, V_p, nut_p))

            J, h, c, d = niw.expected_stats(niw_nat)
            E_mniw_params = mniw.expected_stats(mniw_nat)

        ### Get expected potentials from PGM params.
        # NIW
        init = (-2 * J, h)

        # MNIW
        # has mean M = [A|b] so we must break apart matrices into constituent parts
        transition_params = (-jnp.expand_dims(E_mniw_params[2][-1,:-1],-1) - jnp.expand_dims(E_mniw_params[2][:-1,-1],-1),
                             E_mniw_params[2][:-1,:-1] * -2,
                             E_mniw_params[1][:-1,:],
                             E_mniw_params[0] * -2,
                             jnp.expand_dims(E_mniw_params[1][-1,:],-1))
        E_init_normalizer = jnp.log(2 * jnp.pi)*self.latent_D/2 - c - d

        E_trans_normalizer = (x[0].shape[0]-1) * (jnp.log(2 * jnp.pi)*transition_params[2].shape[-1]/2 - (E_mniw_params[2][-1,-1] + E_mniw_params[-1]))
        E_prior_logZ = E_init_normalizer + E_trans_normalizer

        pgm_potentials = init, transition_params
        if self.point_est:
            return pgm_potentials, jnp.zeros(()), E_prior_logZ, None
        global_natparams = niw_nat, mniw_nat
        return pgm_potentials, self.calc_prior_loss(global_natparams), E_prior_logZ, global_natparams

    def __call__(self, recog_potentials, key, n_forecast = 0, n_samples = 1):
        # get expectations of q(theta)
        if n_forecast > 0:
            key, forecast_rng = split(key)

        pgm_potentials, prior_kl, E_prior_logZ, global_natparams = self.expected_params(recog_potentials)
        if self.inf32:
            recog_potentials, pgm_potentials, E_prior_logZ = tree_map(lambda x: x.astype(jnp.float32), (recog_potentials, pgm_potentials, E_prior_logZ))

        # PGM Inference
        if self.drop_correction:
            inference_params = tree_map(lambda x: stop_gradient(x), pgm_potentials)
        else:
            inference_params = pgm_potentials

        gaus_expected_stats, logZ, _ = lds_inference_homog(recog_potentials, *inference_params)

        # Sample z
        if n_samples > 1:
            key = split(key, n_samples)
            z = vmap(sample_lds, in_axes=[None, 0])(gaus_expected_stats, key)
        else:
            z = sample_lds(gaus_expected_stats, key)

        # calculate surrogate loss
        sur_loss = lds_kl_surr(recog_potentials, gaus_expected_stats, E_prior_logZ, logZ)

        # calculate local kl
        if self.drop_correction:
            local_kl = lds_kl_full(recog_potentials, gaus_expected_stats,
                                   *lds_transition_params_to_nat(*pgm_potentials),
                                   *lds_transition_params_to_nat(*inference_params), E_prior_logZ, logZ)
        else:
            local_kl = lds_kl(recog_potentials, gaus_expected_stats, E_prior_logZ, logZ)

        # forecast
        if n_forecast > 0:
            forecasted_z = lds_forecast(z[-1], global_natparams[-1], n_forecast, forecast_rng)
            z = jnp.concatenate([z, forecasted_z], -2)
        return z, (gaus_expected_stats,), prior_kl, local_kl, sur_loss
    
    def iwae(self, recog_potentials, rng, theta_rng, n=1):
        pgm_potentials, _, _, global_natparams = self.expected_params(recog_potentials)
        gaus_expected_stats, logZ, _ = lds_inference_homog(recog_potentials, *pgm_potentials)

        # sample from q(theta) and evaluate kl with the prior
        key, subkey = split(theta_rng)
        niw_sample, niw_global_kl = sample_and_logprob(self.niw_0, global_natparams[0], niw.logZ, 
                                                       partial(niw.sample_es, key=key), n=n)
        mniw_sample, mniw_global_kl = sample_and_logprob(self.mniw_0, global_natparams[1], mniw.logZ, 
                                                       partial(mniw.sample_es, key=subkey), n=n)
        # sample from q(z)
        mapped_rng = split(rng,n)
        zs = vmap(sample_lds, in_axes=[None, 0])(gaus_expected_stats, mapped_rng)

        # get logZ of p(z | theta) for our samples
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

        prior_params, prior_logZs = vmap(get_lds_logZ)(niw_sample, mniw_sample)

        # get difference between p(z|theta) and q(z)
        EXXT = vmap(vmap(lambda x: jnp.outer(x,x)))(zs)
        EX = jnp.expand_dims(zs, -1)
        EXXNT = vmap(vmap(lambda x,y: jnp.outer(x,y)))(zs[:,:-1], zs[:,1:])
        z_ss = (EXXT, EX, EXXNT)
        lds_kl_fun = vmap(lds_kl_full, in_axes=[None, 0, 0, 0, None, None, 0, None])
        lds_kl = lds_kl_fun(recog_potentials, z_ss, *lds_transition_params_to_nat(*prior_params),
                            *lds_transition_params_to_nat(*pgm_potentials), prior_logZs, logZ)
        return zs, niw_global_kl + mniw_global_kl, lds_kl

class SVAE_LDS(Module):
    latent_D: int
    input_D: int
    log_input: bool = False
    encoder_cls: ModuleDef = Encoder
    decoder_cls: ModuleDef = SigmaDecoder
    pgm_hyperparameters: Dict = field(default_factory=dict)
    autoreg: bool = False

    def setup(self):
        self.encoder = self.encoder_cls(self.latent_D, name="encoder")
        self.pgm = PGM_LDS(self.latent_D, name="pgm", **self.pgm_hyperparameters)
        self.decoder = self.decoder_cls(self.input_D, name="decoder")

    @compact
    def __call__(self, x, eval_mode=False, mask=None, n_iwae_samples=0, theta_rng=None, n_forecast = 0, n_samples = 1, fixed_samples = None):

        if self.log_input:
            x = jnp.log(x)

        if not (mask is None):
            unscaled_mask = mask
            mask = jnp.where(mask > 0, jnp.ones_like(mask), jnp.zeros_like(mask))

        x_input = jnp.where(jnp.expand_dims(mask, -1), x, jnp.zeros_like(x)) if mask is not None else x
        recog_potentials = self.encoder(x_input, eval_mode = eval_mode, mask=mask)

        if mask is not None:
            recog_potentials = mask_potentials(recog_potentials, mask)

        key = split(self.make_rng('sampler'),x.shape[0])

        if n_iwae_samples > 0:
            with jax.default_matmul_precision('float32'):
                iwae_fun = vmap(self.pgm.iwae, in_axes=[0,0,None,None])
                z, prior_kl, local_kl = iwae_fun(recog_potentials, key, theta_rng, n_iwae_samples)
            likelihood = self.decoder(z, eval_mode=eval_mode)
            # z will be B x N_iwae_samples x T x D; kl will be B x N_iwae_samples
            return likelihood, prior_kl, local_kl, z

        with jax.default_matmul_precision('float32'):
            pgm_fun = vmap(self.pgm, in_axes=[0,0,None, None])
            z, aux, prior_kl, local_kl, sur_loss = pgm_fun(recog_potentials, key, n_forecast, n_samples)

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

def uton_allparams(state):
    params = state.params['pgm']
    output = {}
    output['mniw'] = mniw.uton((params['St'], params['M'], 
                                params['V'], params['nut']))

    output['niw'] = niw.uton((params['S'], params['loc'],
                              params['lam'], params['nu']))

    return state.replace(params = state.params.copy({'pgm': output}))

def ntou_allparams(state):
    params = state.params['pgm']
    output = {}
    output['St'], output['M'], output['V'], output['nut'] = mniw.ntou(params['mniw'])
    
    output['S'], output['loc'], output['lam'], output['nu'] = niw.ntou(params['niw'])
    return state.replace(params = state.params.copy({'pgm': output}))
