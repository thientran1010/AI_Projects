import unittest
import torch
import os
import tempfile
from DQNAgent import DQNAgent
from hparams import HParams
import gymnasium as gym

class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=float)
        self.spec = type("Spec", (), {"id": "DummyEnv-v0"})()
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return (torch.zeros(4), {})

    def step(self, action):
        self.step_count += 1
        # End the episode after 3 steps
        terminated = self.step_count >= 3
        return (torch.zeros(4), 1.0, terminated, False, {})

class TestDQNAgentTrain(unittest.TestCase):

    def setUp(self):
        self.env = DummyEnv()
        self.hparams = HParams(n_neurons=16, n_layers=1, num_episodes=1)
        self.agent = DQNAgent(env=self.env, hparams=self.hparams)

    def test_train_runs_successfully(self):
        self.agent.train(num_episodes=1)
        self.assertGreaterEqual(len(self.agent.episode_durations), 1, "Training did not record episode durations.")

if __name__ == "__main__":
    unittest.main()
