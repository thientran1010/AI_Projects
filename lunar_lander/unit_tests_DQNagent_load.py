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
        self.spec = type('Spec', (), {'id': 'DummyEnv-v0'})()

    def reset(self, **kwargs):
        return (torch.zeros(4), {})

    def step(self, action):
        return (torch.zeros(4), 1.0, False, False, {})

class TestDQNAgentLoad(unittest.TestCase):

    def setUp(self):
        self.env = DummyEnv()
        self.hparams = HParams(n_neurons=16, n_layers=1, num_episodes=1)

    def test_load_restores_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pth")

            # Original agent
            agent = DQNAgent(env=self.env, hparams=self.hparams)

            # Manually modify weights
            for param in agent.policy_net.parameters():
                param.data.fill_(0.42)  # fill all weights with known value

            agent.save(filepath)

            # New agent
            new_agent = DQNAgent(env=self.env, hparams=self.hparams)
            new_agent.load(filepath)

            # Check weights were restored
            for param in new_agent.policy_net.parameters():
                self.assertTrue(torch.all(param.data == 0.42), "Weights were not restored correctly")

if __name__ == "__main__":
    unittest.main()
