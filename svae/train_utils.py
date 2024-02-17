from jax import tree_map, jit, value_and_grad, vmap
from jax.random import split
from jax.numpy import ones, where, expand_dims, log
from typing import Callable, Any
from flax.core import FrozenDict
from flax.struct import PyTreeNode, field
from functools import partial
import optax
import pickle
import flax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from jax.experimental.host_callback import id_print
from jax.scipy.special import logsumexp
from utils import wandb_log_internal
import jax 

class TrainState(PyTreeNode):
    """
    Expanded reimplementaion of Flax's TrainState class. Encapsulates model parameters, 
    the main model function used in training and the optimization state. 

    For VAE support, also retains the current state of the RNGs using in latent sampling, 
    updating after each gradient application. 

    Inheriting from flax.struct.PytreeNode is a convinient way to register class as a
    JAX PyTree, which allows it to be the input/output of complied JAX functions.
    """
    step: int
    apply_fn: Callable = field(pytree_node=False) # Speify this field as a normal python class (not serializable pytree)
    params: FrozenDict[str, Any]
    batch_stats: FrozenDict[str, Any]
    rng_state: FrozenDict[str, Any]
    tx: optax.GradientTransformation = field(pytree_node=False)
    opt_state: optax.OptState

    def update_rng(self):
        """Update the state of all RNGs stored by the trainstate and return a key for use in this sampling step"""
        new_key, sub_key = tree_map(lambda r: split(r)[0], self.rng_state), tree_map(lambda r: split(r)[1], self.rng_state)
        return self.replace(rng_state=new_key), sub_key

    def apply_gradients(self, *, grads, batch_stats, **kwargs):
        """Take a gradient descent step with the specified grads and encapsulated optimizer."""
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            batch_stats=batch_stats,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, batch_stats, rng_state, tx,  **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            batch_stats=batch_stats,
            rng_state=rng_state,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

def create_train_state(rng, learning_rate, model, input_shape):
    """Creates initial `TrainState`."""
    model = model()
    init_rng, init_model_rng, model_rng = split(rng, 3)
    init = model.init({'params': init_rng, 'sampler': init_model_rng}, ones(input_shape))
    tx = optax.adam(learning_rate)
    return model, TrainState.create(
        apply_fn=model.apply, params=init['params'], batch_stats=init['batch_stats'] if 'batch_stats' in init else flax.core.FrozenDict(),
        rng_state=FrozenDict(sampler=model_rng), tx=tx)

def neg_log_lik_loss(dist, x, mask = None):
    """Negative log-likelihood of an observation"""
    if mask is not None:
        probs = where(expand_dims(mask, -1), -dist.log_prob(x), 0)
        return probs.sum()
    return -dist.log_prob(x).sum()

@partial(jit, static_argnums=6)
def train_step(state, batch, mask=None, N_data = 1, local_kl_weight=1., prior_kl_weight=1., log_magnitudes=False, **kwargs):
    """Train for a single step."""
    def loss_fn(params, rngs, batch_stats):
        """Inner function that computes loss and metrics."""
        (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                                    batch, mask=mask, rngs=rngs, mutable=['batch_stats'], **kwargs)
        recon_loss = neg_log_lik_loss(likelihood, batch, mask=mask)
        if batch.ndim == 3:
            recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]
        loss = recon_loss + prior_kl * prior_kl_weight + local_kl * local_kl_weight
        return loss, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, loss=recon_loss + prior_kl + local_kl, batch_stats=batch_stats['batch_stats'], aux=aux)

    # Create a function that runs loss_fn and gets the gradient with respect to the first input.
    grad_fn = value_and_grad(loss_fn, has_aux=True)

    # Run the transformed function and apply the resulting gradients to the state
    state, keys = state.update_rng()
    (_, outputs), grads = grad_fn(state.params, keys, state.batch_stats)
    
    # Log gradient magnitudes
    if log_magnitudes:
        pg = 0.
        if 'pgm' in grads:
            pg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['pgm'])[0] ** 2))
        eg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['encoder'])[0] ** 2))
        dg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['decoder'])[0] ** 2))
        props = jnp.mean(outputs['aux'][-1], axis=(0, 1))
        max_used_state = jnp.max(props)
        n_used_states = jnp.sum(props > (1. / (props.shape[0] * 2)))
        jax.debug.callback(lambda pgm_g, encoder_g, decoder_g, mus, nus: wandb_log_internal(dict(pgm_grad_norm=pgm_g, encoder_grad_norm=encoder_g, decoder_grad_norm=decoder_g, max_used_state=mus, n_used_states=nus)), pg, eg, dg, max_used_state, n_used_states) 
    
    state = state.apply_gradients(grads=grads, batch_stats=outputs['batch_stats'])
    metrics = dict(recon_loss=outputs['recon_loss'], prior_kl=outputs['prior_kl'], local_kl = outputs['local_kl'], loss=outputs['recon_loss'] + outputs['prior_kl'] + outputs['local_kl'], aux=outputs['aux'][-1])

    # Return the updated state and computed metrics
    return state, metrics

def make_train_step_with_transform(trans):
    @partial(jit, static_argnums=6)
    def train_step_with_transform(state, batch, mask=None, N_data = 1, local_kl_weight=1., prior_kl_weight=1., log_magnitudes=False, **kwargs):
        """Train for a single step."""
        def loss_fn(params, rngs, batch_stats):
            """Inner function that computes loss and metrics."""
            (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn({'params': params, 
                                                                                 'batch_stats': state.batch_stats},
                                                                                trans(batch), mask=mask, rngs=rngs, 
                                                                                mutable=['batch_stats'], **kwargs)
            recon_loss = neg_log_lik_loss(likelihood, batch, mask=mask)
            if batch.ndim == 3:
                recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]
            loss = recon_loss + prior_kl * prior_kl_weight + local_kl * local_kl_weight
            return loss, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, loss=recon_loss + prior_kl + local_kl, batch_stats=batch_stats['batch_stats'], aux=aux)

        # Create a function that runs loss_fn and gets the gradient with respect to the first input.
        grad_fn = value_and_grad(loss_fn, has_aux=True)

        # Run the transformed function and apply the resulting gradients to the state
        state, keys = state.update_rng()
        (_, outputs), grads = grad_fn(state.params, keys, state.batch_stats)

        # Log gradient magnitudes
        if log_magnitudes:
            pg = 0.
            if 'pgm' in grads:
                pg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['pgm'])[0] ** 2))
            eg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['encoder'])[0] ** 2))
            dg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['decoder'])[0] ** 2))
            props = jnp.mean(outputs['aux'][-1], axis=(0, 1))
            max_used_state = jnp.max(props)
            n_used_states = jnp.sum(props > (1. / (props.shape[0] * 2)))
            jax.debug.callback(lambda pgm_g, encoder_g, decoder_g, mus, nus: wandb_log_internal(dict(pgm_grad_norm=pgm_g, encoder_grad_norm=encoder_g, decoder_grad_norm=decoder_g, max_used_state=mus, n_used_states=nus)), pg, eg, dg, max_used_state, n_used_states) 

        state = state.apply_gradients(grads=grads, batch_stats=outputs['batch_stats'])
        metrics = dict(recon_loss=outputs['recon_loss'], prior_kl=outputs['prior_kl'], local_kl = outputs['local_kl'], loss=outputs['recon_loss'] + outputs['prior_kl'] + outputs['local_kl'], aux=outputs['aux'][-1])

        # Return the updated state and computed metrics
        return state, metrics
    return train_step_with_transform

def make_eval_step_with_transform(trans):
    @jit
    def eval_step_with_transform(state, batch, mask=None, N_data=1, **kwargs):
        """Compute metrics for a single batch. Still returns updated state to account for consuming RNG state."""
        state, keys = state.update_rng()
        likelihood, prior_kl, local_kl, aux = state.apply_fn({'params': state.params, "batch_stats": state.batch_stats},
                                                          trans(batch), eval_mode=True, mask=mask, rngs=keys, **kwargs)
        recon_loss = neg_log_lik_loss(likelihood, batch, mask=mask)
        if batch.ndim == 3: #Check this
            recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]

        loss = recon_loss + prior_kl + local_kl
        return state, loss, likelihood, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, likelihood=likelihood.log_prob(batch), loss=loss, aux=aux)
    return eval_step_with_transform

@partial(jit, static_argnums=6)
def grad_step(state, batch, mask=None, N_data = 1, local_kl_weight=1., prior_kl_weight=1., log_magnitudes=True, **kwargs):
    """Train for a single step."""
    def loss_fn(params, rngs, batch_stats):
        """Inner function that computes loss and metrics."""
        (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                                    batch, mask=mask, rngs=rngs, mutable=['batch_stats'], **kwargs)
        recon_loss = neg_log_lik_loss(likelihood, batch, mask=mask)
        if batch.ndim == 3:
            recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]
        loss = recon_loss + prior_kl * prior_kl_weight + local_kl * local_kl_weight
        return loss, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, loss=loss, batch_stats=batch_stats['batch_stats'], aux=aux)

    # Create a function that runs loss_fn and gets the gradient with respect to the first input.
    grad_fn = value_and_grad(loss_fn, has_aux=True)

    # Run the transformed function and apply the resulting gradients to the state
    state, keys = state.update_rng()
    (_, outputs), grads = grad_fn(state.params, keys, state.batch_stats)
    
    return grads, outputs

@partial(jit, static_argnums=2)
def train_step_multisample(state, batch, n_samples, mask=None, N_data = 1, local_kl_weight=1., prior_kl_weight=1., log_magnitudes=False, **kwargs):
    """Train for a single step."""
    def loss_fn(params, rngs, batch_stats):
        """Inner function that computes loss and metrics."""
        (likelihood, prior_kl, local_kl, aux), batch_stats = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                                    batch, mask=mask, n_samples=n_samples, rngs=rngs, mutable=['batch_stats'], **kwargs)
        recon_loss = -likelihood.log_prob(expand_dims(batch,0)).mean(axis=[0]).sum()
        if batch.ndim == 3:
            recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]
        loss = recon_loss + prior_kl * prior_kl_weight + local_kl * local_kl_weight
        return loss, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, loss=loss, batch_stats=batch_stats['batch_stats'])

    # Create a function that runs loss_fn and gets the gradient with respect to the first input.
    grad_fn = value_and_grad(loss_fn, has_aux=True)

    # Run the transformed function and apply the resulting gradients to the state
    state, keys = state.update_rng()
    (_, outputs), grads = grad_fn(state.params, keys, state.batch_stats)
    
    # Log gradient magnitudes
    # Log gradient magnitudes
    if log_magnitudes:
        pg = 0.
        if 'pgm' in grads:
            pg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['pgm'])[0] ** 2))
        eg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['encoder'])[0] ** 2))
        dg = jnp.sqrt(jnp.mean(jax.flatten_util.ravel_pytree(grads['decoder'])[0] ** 2))
        props = jnp.mean(outputs['aux'][-1], axis=(0, 1))
        max_used_state = jnp.max(props)
        n_used_states = jnp.sum(props > (1. / (props.shape[0] * 2)))
        jax.debug.callback(lambda pgm_g, encoder_g, decoder_g, mus, nus: wandb_log_internal(dict(pgm_grad_norm=pgm_g, encoder_grad_norm=encoder_g, decoder_grad_norm=decoder_g, max_used_state=mus, n_used_states=nus)), pg, eg, dg, max_used_state, n_used_states) 
    
    state = state.apply_gradients(grads=grads, batch_stats=outputs['batch_stats'])
    metrics = dict(recon_loss=outputs['recon_loss'], prior_kl=outputs['prior_kl'], local_kl = outputs['local_kl'], loss=outputs['loss'])

    # Return the updated state and computed metrics
    return state, metrics

@jit
def eval_step(state, batch, mask=None, N_data=1, **kwargs):
    """Compute metrics for a single batch. Still returns updated state to account for consuming RNG state."""
    state, keys = state.update_rng()
    likelihood, prior_kl, local_kl, aux = state.apply_fn({'params': state.params, "batch_stats": state.batch_stats},
                                                      batch, eval_mode=True, mask=mask, rngs=keys, **kwargs)
    recon_loss = neg_log_lik_loss(likelihood, batch, mask=mask)
    if batch.ndim == 3: #Check this
        recon_loss, prior_kl, local_kl = recon_loss/batch.shape[0], prior_kl/N_data, local_kl/batch.shape[0]

    loss = recon_loss + prior_kl + local_kl
    return state, loss, likelihood, dict(recon_loss=recon_loss, prior_kl=prior_kl, local_kl=local_kl, likelihood=likelihood.log_prob(batch), loss=loss, aux=aux)

@partial(jit, static_argnums=2)
def eval_step_iwae(state, batch, n_iwae_samples, theta_rng, **kwargs):
    state, keys = state.update_rng()
    likelihood, prior_kl, local_kl, z = state.apply_fn({'params': state.params, 
                                                        "batch_stats": state.batch_stats},
                                                       batch, eval_mode=True, rngs=keys, 
                                                       n_iwae_samples=n_iwae_samples, 
                                                       theta_rng=theta_rng, **kwargs)
    recon_loss = -likelihood.log_prob(expand_dims(batch,1)).sum(axis=[-2,-1])
    if batch.ndim == 3:
        prior_kl = prior_kl.mean(0)
        loss = (local_kl + recon_loss).sum(0)
    else:
        loss = local_kl + recon_loss
    return state, prior_kl, loss # need to logsumexp and subtract log(n)

@partial(jit, static_argnums=2)
def eval_step_forecast(state, batch, n_forecast, mask=None, **kwargs):
    state, keys = state.update_rng()
    likelihood, prior_kl, local_kl, aux = state.apply_fn({'params': state.params, 
                                                          "batch_stats": state.batch_stats},
                                                         batch, eval_mode=True, mask=mask, rngs=keys, 
                                                         n_forecast=n_forecast, **kwargs)
    return state, None, likelihood, aux

def eval_step_tf_impute(state, batch, sample_rng, mask=None, N_data=1, **kwargs):
    mask = np.array(mask).astype(int)
    fill_mask = np.zeros_like(mask)
    fill_batch = np.array(batch)
    
    for step in range(batch.shape[-2]):
        fill_mask[..., :step] = 1
        fill_mask[..., step] = mask[..., step]
        state, loss, likelihood, aux = eval_step(state, fill_batch, mask=fill_mask, N_data=N_data, **kwargs)
        
        new_rng, sample_rng = jax.random.split(sample_rng)
        sample = np.array(likelihood.sample(seed=new_rng))
        fill_batch[..., step] = fill_batch * mask[..., step] + sample * (1 - mask[..., step])
    return fill_batch
        
### For separate optimization of network and pgm parameters.
class DualTrainState(PyTreeNode):
    step: int
    apply_fn: Callable = field(pytree_node=False) # Speify this field as a normal python class (not serializable pytree)
    params: FrozenDict[str, Any]
    batch_stats: FrozenDict[str, Any]
    rng_state: FrozenDict[str, Any]
    tx_net: optax.GradientTransformation = field(pytree_node=False)
    tx_pgm: optax.GradientTransformation = field(pytree_node=False)
    opt_state_net: optax.OptState
    opt_state_pgm: optax.OptState

    def update_rng(self):
        """Update the state of all RNGs stored by the trainstate and return a key for use in this sampling step"""
        new_key, sub_key = tree_map(lambda r: split(r)[0], self.rng_state), tree_map(lambda r: split(r)[1], self.rng_state)
        return self.replace(rng_state=new_key), sub_key

    def apply_gradients(self, *, grads, batch_stats, **kwargs):
        """Take a gradient descent step with the specified grads and encapsulated optimizer."""
        net_grads, pgm_grads = grads, grads.pop('pgm')
        net_params, pgm_params = self.params, self.params.pop('pgm')
        if self.tx_net is None:
            new_opt_state_net, new_params = None, net_params
        else:
            net_updates, new_opt_state_net = self.tx_net.update(net_grads, self.opt_state_net, net_params)
            new_params = optax.apply_updates(net_params, net_updates)
        if self.tx_pgm is None:
            new_opt_state_pgm, new_params_pgm = None, pgm_params
        else:
            pgm_updates, new_opt_state_pgm = self.tx_pgm.update(pgm_grads, self.opt_state_pgm, pgm_params)
            new_params_pgm = optax.apply_updates(pgm_params, pgm_updates)
        new_params.update({"pgm": new_params_pgm})
        return self.replace(
            step=self.step + 1,
            params=new_params,
            batch_stats=batch_stats,
            opt_state_net=new_opt_state_net,
            opt_state_pgm=new_opt_state_pgm,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, batch_stats, rng_state, tx_net, tx_pgm, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        net_params, pgm_params = params, params.pop('pgm')
        if tx_net is None:
            opt_state_net = None
        else:
            opt_state_net = tx_net.init(net_params)
        if tx_pgm is None:
            opt_state_net = None
        else:
            opt_state_pgm = tx_pgm.init(pgm_params)
        params.update({'pgm': pgm_params})
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            batch_stats=batch_stats,
            rng_state=rng_state,
            tx_net=tx_net,
            tx_pgm = tx_pgm,
            opt_state_net=opt_state_net,
            opt_state_pgm=opt_state_pgm,
            **kwargs,
        )

def create_dual_train_state(rng, learning_rate_net, learning_rate_pgm, model, input_shape, network_params=None, batch_stats=None, learning_alg_pgm='adam', learning_alg_net='adam'):
    model = model()
    init_rng, init_model_rng, model_rng = split(rng, 3)
    init = model.init({'params': init_rng, 'sampler': init_model_rng}, ones(input_shape))
    
    learning_algs = dict(adam=optax.adam, sgd=optax.sgd)
    if learning_rate_net is None:
        tx_net = None
    else:
        tx_net = learning_algs[learning_alg_net](learning_rate_net) 
    if learning_rate_pgm is None:
        tx_pgm = None
    else:
        tx_pgm = learning_algs[learning_alg_pgm](learning_rate_pgm)
    if network_params is None:
        params = init['params']
    else:
        network_params.update({"pgm": init['params']['pgm']})
        if init['params'].get('vmp') is not None:
            network_params.update({"vmp": init['params']['vmp']})
        params = network_params
    if batch_stats is None:
        batch_stats = init['batch_stats'] if 'batch_stats' in init else flax.core.FrozenDict()
    return model, DualTrainState.create(
        apply_fn=model.apply, params=params, batch_stats=batch_stats,
        rng_state=FrozenDict(sampler=model_rng), tx_net=tx_net, tx_pgm = tx_pgm)

def save_state(state, filename):
    if isinstance(state, TrainState):
        with open(filename, 'wb') as f:
            pickle.dump((state.params, state.batch_stats, state.rng_state, state.opt_state), f)
    elif isinstance(state, DualTrainState):
        with open(filename, 'wb') as f:
            pickle.dump((state.params, state.batch_stats, state.rng_state, state.opt_state_net, state.opt_state_pgm), f)
    else:
        print("Invalid state type")

def load_state(state, filename):
    if isinstance(state, TrainState):
        with open(filename, 'rb') as f:
            params, batch_stats, rng_state, opt_state = pickle.load(f)
            return state.replace(params=params,
                                 batch_stats=batch_stats,
                                 rng_state=rng_state,
                                 opt_state=opt_state)
    elif isinstance(state, DualTrainState):
        with open(filename, 'rb') as f:
            params, batch_stats, rng_state, opt_state_net, opt_state_pgm = pickle.load(f)
            return state.replace(params=params,
                                 batch_stats=batch_stats,
                                 rng_state=rng_state,
                                 opt_state_net=opt_state_net,
                                 opt_state_pgm=opt_state_pgm)
    else:
        print("Invalid state type")
        
def save_params(state, filename):
    with open(filename, 'wb') as f:
        pickle.dump((state.params, state.batch_stats), f)

def load_params(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

    
def bind_state(model, state):
    return model.bind({'params': state.params, 'batch_stats': state.batch_stats}, rngs=state.rng_state)