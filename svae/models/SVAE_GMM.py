from jax import tree_map, vmap
from jax.lax import stop_gradient
from jax.random import split
import jax.numpy as jnp
from flax.linen import compact, initializers, softplus, Module
from distributions import mniw, niw, dirichlet, normal
from utils import softminus, T, make_prior_fun, mask_potentials, straight_through, straight_through_tuple, corr_param_inv, inject_mingrads_pd
from typing import Callable, Any, Dict
ModuleDef = Any
from dataclasses import field
from models.VAE import Encoder, Decoder
from inference.GMM_Inference import gmm_inference_itersolve, gmm_kl, gmm_kl_det
import jax

class PGM_GMM(Module):
    latent_D: int
    K: int
    inference_fun: Callable
    nat_grads: bool = False
    new_nat_grads: bool = True
    drop_correction: bool = False
    kappa_0: float = 0.01
    S_0: float = 1.
    nu_0: float = 2.
    lam_0: float = 0.001
    kappa_init: float = 1.
    S_init: float = 1.
    nu_init: float = 2.
    lam_init: float = 20.
    loc_init_sd: float = 1. 
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

    def calc_prior_loss(self, niw_p, kappa_p):
        return jnp.sum(self.niw_prior_kl(niw_p)) + self.kappa_prior_kl(kappa_p)

    @compact
    def expected_params(self):
        ### Initializations and converting from unconstrained space
        S = jnp.tile(corr_param_inv(jnp.identity(self.latent_D) * self.latent_D * self.S_init), (self.K,1,1))
        def gen_loc(key):
            return jax.random.normal(key, (self.K, self.latent_D, 1)) * self.loc_init_sd

        lam = jnp.ones(self.K) * softminus(self.lam_init)
        nu = jnp.ones(self.K) * softminus(self.nu_init + 1.)

        kappa = jnp.ones(self.K) * softminus(self.kappa_init)

        if self.nat_grads:
            niw_nat = self.param("niw", lambda rng: vmap(niw.uton)((S, gen_loc(rng), lam, nu)))

            kappa_nat = self.param("kappa", lambda rng: dirichlet.uton(kappa))

            niw_nat = (vmap(inject_mingrads_pd)(niw_nat[0]), 
                       niw_nat[1], niw_nat[2], niw_nat[3])

            # calculate expected statistics
            J, h, c, d = vmap(straight_through_tuple(niw.expected_stats))(niw_nat)
            cat_params = straight_through(dirichlet.expected_stats)(kappa_nat)

        elif self.new_nat_grads:
            S_p = self.param("S", lambda rng: S)
            loc_p = self.param("loc", lambda rng: gen_loc(rng))
            lam_p = self.param("lam", lambda rng: lam)
            nu_p = self.param("nu", lambda rng: nu)

            niw_nat = vmap(niw.uton_natgrad)((S_p, loc_p, lam_p, nu_p))
            niw_nat = (vmap(inject_mingrads_pd)(niw_nat[0]), 
                       niw_nat[1], niw_nat[2], niw_nat[3])

            kappa_p = self.param("kappa", lambda rng: kappa)
            kappa_nat = dirichlet.uton_natgrad(kappa_p)

            J, h, c, d = vmap(straight_through_tuple(niw.expected_stats))(niw_nat)
            cat_params = straight_through(dirichlet.expected_stats)(kappa_nat)

        else:
            S_p = self.param("S", lambda rng: S)
            loc_p = self.param("loc", lambda rng: gen_loc(rng))
            lam_p = self.param("lam", lambda rng: lam)
            nu_p = self.param("nu", lambda rng: nu)

            niw_nat = vmap(niw.uton)((S_p, loc_p, lam_p, nu_p))

            kappa_p = self.param("kappa", lambda rng: kappa)
            kappa_nat = dirichlet.uton(kappa_p)

            J, h, c, d = vmap(niw.expected_stats)(niw_nat)
            cat_params = dirichlet.expected_stats(kappa_nat)

        gaus_params = (J, h)
        gaus_normalizer = jnp.log(2 * jnp.pi)*self.latent_D/2 - c - d

        pgm_potentials = gaus_params, gaus_normalizer, cat_params
        global_natparams = niw_nat, kappa_nat
        return pgm_potentials, self.calc_prior_loss(niw_nat, kappa_nat), global_natparams

    def __call__(self, recog_potentials, key, initializer):

        # Get expectations of q(theta)
        pgm_potentials, prior_kl, global_natparams = self.expected_params()
        if self.inf32:
            recog_potentials, pgm_potentials = tree_map(lambda x: x.astype(jnp.float32), (recog_potentials, pgm_potentials))

        # PGM Inference
        if self.drop_correction:
            inference_params = tree_map(lambda x: stop_gradient(x), pgm_potentials)
        else:
            inference_params = pgm_potentials

        gaus_expected_stats, cat_expected_stats = self.inference_fun(recog_potentials,  *inference_params, initializer)

        if self.drop_correction:
            local_kl = gmm_kl_det(recog_potentials, pgm_potentials, inference_params,
                                  gaus_expected_stats, cat_expected_stats)
        else:
            local_kl = gmm_kl(recog_potentials, *pgm_potentials, 
                              gaus_expected_stats, cat_expected_stats)

        z = normal.sample_from_es(gaus_expected_stats, key)[0]
        return z, cat_expected_stats, prior_kl, local_kl

class SVAE_GMM(Module):
    latent_D: int
    K: int
    input_D: int
    encoder_cls: ModuleDef = Encoder
    decoder_cls: ModuleDef = Decoder
    inference_fun: Callable = gmm_inference_itersolve
    pgm_hyperparameters: Dict = field(default_factory=dict)

    def setup(self):
        self.encoder = self.encoder_cls(self.latent_D, name="encoder")
        self.pgm = PGM_GMM(self.latent_D, self.K, self.inference_fun, name="pgm", **self.pgm_hyperparameters)
        self.decoder = self.decoder_cls(self.input_D, name="decoder")

    @compact
    def __call__(self, x, eval_mode=False, mask=None, initializer = None):
        
        if not (mask is None):
            raise NotImplementedError

        recog_potentials = self.encoder(x, eval_mode = eval_mode, mask=mask)

        if mask is not None:
            recog_potentials = mask_potentials(recog_potentials, mask)

        key = self.make_rng('sampler')
        if initializer is None:
            key, initializer = split(key)

        with jax.default_matmul_precision('float32'):
            z, aux, prior_kl, local_kl = self.pgm(recog_potentials, key, initializer)

        likelihood = self.decoder(z.astype(jnp.float32), eval_mode=eval_mode)
        return likelihood, prior_kl, local_kl, (z, aux)
