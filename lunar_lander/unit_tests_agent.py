import unittest
import DQNAgent
import gymnasium as gym
import dqnModel

class test_replayMemory(unittest.TestCase):
    def test_init_trivial(self):
        test_replay = DQNAgent.ReplayMemory(5)
        self.assertEqual = (5, test_replay.memory.maxlen)

    def test_init_lower_limit(self):
        with self.assertRaises(ValueError):
            test_replay = DQNAgent.ReplayMemory(-5)

    def test_push_wrong_tuple_size(self):
        test_replay = DQNAgent.ReplayMemory(5)
        with self.assertRaises(TypeError):
            test_replay.push(3)
    


class test_DQNAgent(unittest.TestCase):
    def setUp(self):
          self.agent = DQNAgent.DQNAgent(gym.make("LunarLander-v3",render_mode="human"), odel_class=r"dqnModel_lunar_lander.pt")

    def test_agent_init(self):
        self.assertIsInstance(self.agent.policy_net, dqnModel)
        self.assertIsInstance(self.agent.target_net, dqnModel)

if __name__ == '__main__':
    unittest.main()