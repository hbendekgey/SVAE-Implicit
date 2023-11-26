# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A text classification model."""

import functools
from typing import Any, Callable, Optional

from flax import linen as nn
import jax
from jax import numpy as jnp

Array = jnp.ndarray


@jax.vmap
def flip_sequences(inputs: Array, lengths: Array = None) -> Array:
    """Flips a sequence of inputs along the time dimension.

    This function can be used to prepare inputs for the reverse direction of a
    bidirectional LSTM. It solves the issue that, when naively flipping multiple
    padded sequences stored in a matrix, the first elements would be padding
    values for those sequences that were padded. This function keeps the padding
    at the end, while flipping the rest of the elements.

    Example:
    ```python
    inputs = [[1, 0, 0],
            [2, 3, 0]
            [4, 5, 6]]
    lengths = [1, 2, 3]
    flip_sequences(inputs, lengths) = [[1, 0, 0],
                                     [3, 2, 0],
                                     [6, 5, 4]]
    ```

    Args:
    inputs: An array of input IDs <int>[batch_size, seq_length].
    lengths: The length of each sequence <int>[batch_size].

    Returns:
    An ndarray with the flipped inputs.
    """
    # Note: since this function is vmapped, the code below is effectively for
    # a single example.
    max_length = inputs.shape[0]
    if lengths is None:
        lengths = max_length
    return jnp.flip(jnp.roll(inputs, max_length - lengths, axis=0), axis=0)


class SimpleLSTM(nn.Module):
    """A simple unidirectional LSTM."""

    @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1,
      split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        x, mask, initial_state = x
        new_carry, output = nn.OptimizedLSTMCell(dtype=jnp.float32, param_dtype=jnp.float32)(carry, x)
        new_carry = jax.tree_map(lambda x, y: jnp.where(mask, x, y), new_carry, initial_state)
        output = jnp.where(mask, output, jnp.zeros_like(output))
        return new_carry, output

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
        # Use fixed random key since default state init fn is just zeros.
        return jax.tree_map(lambda x: x.astype(jnp.float32), nn.OptimizedLSTMCell.initialize_carry(
            jax.random.PRNGKey(0), batch_dims, hidden_size))
    
class ReverseSimpleLSTM(nn.Module):
    """A simple unidirectional LSTM."""

    @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=1, out_axes=1, reverse=True,
      split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        x, mask, initial_state = x
        new_carry, output = nn.OptimizedLSTMCell(dtype=jnp.float32, param_dtype=jnp.float32)(carry, x)
        new_carry = jax.tree_map(lambda x, y: jnp.where(mask, x, y), new_carry, initial_state)
        output = jnp.where(mask, output, jnp.zeros_like(output))
        return new_carry, output

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
        # Use fixed random key since default state init fn is just zeros.
        return jax.tree_map(lambda x: x.astype(jnp.float32), nn.OptimizedLSTMCell.initialize_carry(
            jax.random.PRNGKey(0), batch_dims, hidden_size))
    
    
class LSTM(nn.Module):
    """A simple bi-directional LSTM."""
    hidden_size: int
    eval_mode: bool = False

    def setup(self):
        self.forward_lstm = SimpleLSTM()

    def __call__(self, embedded_inputs, lengths=None, mask=None):
        batch_size = embedded_inputs.shape[0]
        mask = mask if not (mask is None) else jnp.ones_like(embedded_inputs[..., 0])
        mask = jnp.expand_dims(mask, -1)
        initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
        carry_input = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 1), embedded_inputs.shape[1], axis=1), initial_state) 
        _, forward_outputs = self.forward_lstm(initial_state, (embedded_inputs, mask, carry_input))
        return forward_outputs
    
class ReverseLSTM(nn.Module):
    """A simple bi-directional LSTM."""
    hidden_size: int
    eval_mode: bool = False

    def setup(self):
        self.backward_lstm = SimpleLSTM()

    def __call__(self, embedded_inputs, lengths=None, mask=None):
        batch_size = embedded_inputs.shape[0]
        reversed_inputs = flip_sequences(embedded_inputs, lengths)
        reversed_masks = flip_sequences(mask, lengths) if not (mask is None) else jnp.ones_like(reversed_inputs[..., 0])
        reversed_masks = jnp.expand_dims(reversed_masks, -1)   
        initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
        carry_input = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 1), embedded_inputs.shape[1], axis=1), initial_state)
        _, backward_outputs = self.backward_lstm(initial_state, (reversed_inputs, reversed_masks, carry_input))
        backward_outputs = flip_sequences(backward_outputs, lengths)

        return backward_outputs


class SimpleBiLSTM(nn.Module):
    """A simple bi-directional LSTM."""
    hidden_size: int
    eval_mode: bool = False 

    def setup(self):
        self.forward_lstm = SimpleLSTM()
        self.backward_lstm = SimpleLSTM()

    def __call__(self, embedded_inputs, lengths=None):
        batch_size = embedded_inputs.shape[0]

        # Forward LSTM.
        initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
        _, forward_outputs = self.forward_lstm(initial_state, embedded_inputs)

        # Backward LSTM.
        reversed_inputs = flip_sequences(embedded_inputs, lengths)
        initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
        _, backward_outputs = self.backward_lstm(initial_state, reversed_inputs)
        backward_outputs = flip_sequences(backward_outputs, lengths)

        # Concatenate the forward and backward representations.
        outputs = jnp.concatenate([forward_outputs, backward_outputs], -1)
        return outputs

