"""
hparams.py

Defines the HParams class to manage hyperparameters for the DQN model,
including number of neurons, layers, and training episodes.

Author: AF
Created: 28-06-2025

Classes:
    - HParams: Provides getter and setter methods for DQN training configuration.
"""
class HParams:

    """
        Set hyperparameters for the DQN model from user.

        @param n_neurons: Number of neurons in each hidden layer.
        @param n_layers: Number of hidden layers.
        @param num_episodes: Number of episodes to train for.

        return: None
    """
    def __init__(self, n_neurons=128, n_layers=1, num_episodes=50):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.num_episodes = num_episodes

    def set_n_neurons(self, n):
        self.n_neurons = n

    def set_n_layers(self, n):
        self.n_layers = n

    def set_num_episodes(self, n):
        self.num_episodes = n

    def get_n_neurons(self):
        return self.n_neurons

    def get_n_layers(self):
        return self.n_layers

    def get_num_episodes(self):
        return self.num_episodes
