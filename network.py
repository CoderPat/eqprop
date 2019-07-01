from abc import ABC, abstractmethod
from collections import namedtuple

import jax.numpy as np
from jax import grad, jit, vmap

from utils import vmap_mean

def rho(state):
    return np.clip(state, 0, 1)

class Network(ABC):
    """ Class that represents an Hopfield network """
    def __init__(self, 
                 states, 
                 weights):
        self.states = states
        self.weights = weights

    def clone(self):
        new_states = [state.copy() for state in self.states]
        return self.__class__(new_states, self.weights)

    def batch(self, batch_size):
        batched_states = []
        for state in self.states:
            tiled_shape = (batch_size,) + tuple(1 for _ in state.shape)
            batched_states.append(np.tile(state, tiled_shape))
        
        return self.__class__(batched_states, self.weights)

    @abstractmethod
    def connected(self, state_idx):
        pass

    def __getitem__(self, i):
        if i == 0 : return self.states
        if i == 1 : return self.weights
        raise IndexError("")

    @classmethod
    def free_dynamics_fn(network_cls):
        """ Returns and jax-compatible energy function specific to the network class """

        def _state_dynamics_fn(states, weights):
            network = network_cls(states, weights)
            dynamics = []
            for i, state in enumerate(network.states):
                incoming = np.zeros_like(state)
                for pre_idx, weight_idx in network.connected(i):
                    incoming = incoming + np.dot(rho(network.states[pre_idx]), network.weights[weight_idx])
                dynamics.append(incoming - state)

            #print("output dynamics {}".format(dynamics[-1]))

            return dynamics

        return _state_dynamics_fn

    @classmethod
    def cost_fn(network_cls):
        """ Returns and jax-compatible cost function specific to the network class """

        def _cost_fn(states, weights, y):
            network = network_cls(states, weights)
            return np.sum((y - network.output)**2)

        return _cost_fn


    @classmethod
    def clamped_dynamics_fn(network_cls):
        """ Returns and jax-compatible total function specific to the network class """

        free_dynamics_fn = network_cls.free_dynamics_fn()
        cost_fn = network_cls.cost_fn()
        cost_grad = grad(cost_fn)

        def _clamped_dynamics_fn(states, weights, y, beta):
            free_dynamics = free_dynamics_fn(states, weights)
            grads = cost_grad(states, weights, y)
            return [mu - beta * g for mu, g in zip(free_dynamics, grads)]

        return _clamped_dynamics_fn



def free_relaxation(network, num_steps=100, epsilon=0.01):
    """ TODO """
    free_dynamics = network.free_dynamics_fn()
    # move states in direction that reduces the energy
    for _ in range(num_steps):
        for i, delta in enumerate(free_dynamics(*network)[1:], 1):
            network.states[i] = np.clip(network.states[i] + delta * epsilon, 0, 1)
    return network


def clamped_relaxation(network, y, beta=1000., lr=0.05, num_steps=100, epsilon=0.01):
    """ TODO """
    clamped_network = network.clone()
    clamped_dynamics = network.clamped_dynamics_fn()

    # move clamped states in direction that reduces the total energy
    for _ in range(num_steps):
        for i, delta in enumerate(clamped_dynamics(*clamped_network, y, beta)[1:], 1):
            clamped_network.states[i] = np.clip(clamped_network.states[i] + delta * epsilon, 0, 1)

    # calculate difference in states 
    states_diffs = [s2 - s1 for s1, s2 in zip(network.states, clamped_network.states)]

    for state_idx, state_diff in enumerate(states_diffs):
        for pre_idx, weight_idx in network.connected(state_idx):
            grad = np.dot(rho(network.states[pre_idx])[:, np.newaxis], state_diff[np.newaxis, :])
            #print("weights {}".format(network.weights[weight_idx]))
            #print("grads {}".format(grad))
            network.weights[weight_idx] = network.weights[weight_idx] + lr * grad

    return network



# JIT-able versions of the relaxation function, with batching as an option
def jit_free_relaxation(network_cls, num_steps=100, epsilon=0.01, batched=False):
    """ """
    free_dynamics = network_cls.free_dynamics_fn()

    @jit
    def _free_relaxation(states, weights):
        for _ in range(num_steps):
            for i, delta in enumerate(free_dynamics(states, weights)[1:], 1):
                states[i] = np.clip(states[i] + delta * epsilon, 0, 1)

        return states, weights

    if batched:
        _free_relaxation = vmap(_free_relaxation, (0, None), (0, None))
    
    return lambda network: network_cls(*_free_relaxation(*network))


def jit_clamped_relaxation(network_cls, beta=1000., lr=0.05, num_steps=100, epsilon=0.01, batched=False):
    """ """
    clamped_dynamics = network_cls.clamped_dynamics_fn()

    def weight_grads(states, weights, y, beta):
        clamped_states = states.copy()
        for _ in range(num_steps):
            for i, delta in enumerate(clamped_dynamics(states, weights, y, beta)[1:], 1):
                clamped_states[i] = np.clip(clamped_states[i] + delta * epsilon, 0, 1)

        states_diffs = [s2 - s1 for s1, s2 in zip(states, clamped_states)]

        network = network_cls(states, weights)
        grads = [None for _ in range(len(weights))]
        for state_idx, state_diff in enumerate(states_diffs):
            for pre_idx, weight_idx in network.connected(state_idx):
                grad = np.dot(rho(network.states[pre_idx])[:, np.newaxis], state_diff[np.newaxis, :])
                if grads[weight_idx] is None:
                    grads[weight_idx] = np.zeros_like(grad)
                grads[weight_idx] = grads[weight_idx] + grad

        return grads

    if batched:
        weight_grads = vmap_mean(weight_grads, (0, None, 0, None))

    @jit
    def _clamped_relaxation(states, weights, y, beta):
        grads = weight_grads(states, weights, y, beta)
        for weight_idx, grad in enumerate(grads):
            weights[weight_idx] = weights[weight_idx] + lr * grad
        return states, weights

    return lambda network, y: network_cls(*_clamped_relaxation(*network, y, beta))
        

