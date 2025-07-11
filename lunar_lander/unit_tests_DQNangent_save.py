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
        self.spec = type("Spec", (), {"id": "DummyEnv-v0"})()  # ← ADD THIS LINE

    def reset(self, seed=None, options=None):
        return (torch.zeros(4), {})

    def step(self, action):
        return (torch.zeros(4), 1.0, False, False, {})

class TestDQNAgentSave(unittest.TestCase):

    def setUp(self):
        self.env = DummyEnv()
        self.hparams = HParams(n_neurons=16, n_layers=1, num_episodes=1)
        self.agent = DQNAgent(env=self.env, hparams=self.hparams)

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pth")
            self.agent.save(filepath)
            self.assertTrue(os.path.isfile(filepath), "Saved model file does not exist.")

    def test_save_and_load_consistency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pth")
            self.agent.save(filepath)

            new_agent = DQNAgent(env=self.env, hparams=self.hparams)
            new_agent.load(filepath)

            for key in self.agent.policy_net.state_dict():
                self.assertTrue(torch.equal(
                    self.agent.policy_net.state_dict()[key],
                    new_agent.policy_net.state_dict()[key]),
                    f"Mismatch in weights for {key}")

if __name__ == "__main__":
    unittest.main()
