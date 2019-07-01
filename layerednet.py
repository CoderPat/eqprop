import jax.numpy as np
import jax.random as random
from network import Network

from jax.experimental.stax import randn, glorot

#construct states

class LayeredNet(Network):
    @staticmethod
    def new(input_size, 
            output_size, 
            hidden_layers,
            key):

        _randn_fn = randn()
        def vector_init(shape):
            if isinstance(shape, int):
                shape = (shape,)
            nonlocal key
            key, rng = random.split(key)
            return _randn_fn(rng, shape)
        
        _glorot_fn = glorot()
        def matrix_init(shape):
            nonlocal key
            key, rng = random.split(key)
            return _glorot_fn(rng, shape)

        input_state = vector_init(input_size)
        hidden_states = []
        for size in hidden_layers:
            hidden_states.append(vector_init(size))
        output_states = vector_init(output_size)
        states = [input_state, *hidden_states, output_states]

        # weights
        fwd_weights, bwd_weights = [], []
        for prev, post in zip(states[:-1], states[1:]):
            fwd_weights.append(matrix_init((prev.shape[0], post.shape[0])))
            bwd_weights.append(matrix_init((post.shape[0], prev.shape[0])))

        return LayeredNet(states, [*fwd_weights, *bwd_weights])

    @property 
    def x(self):
        return self.states[0]

    @x.setter
    def x(self, value):
        self.states[0] = value

    @property
    def output(self):
        return self.states[-1]

    def connected(self, state_idx):
        if state_idx > 0:
            yield state_idx - 1, state_idx - 1
        if state_idx < len(self.states) - 1:
            yield state_idx + 1, state_idx + len(self.weights) // 2


    
