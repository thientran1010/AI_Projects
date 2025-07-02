import unittest
import DQNAgent
import gymnasium as gym
import dqnModel
import torch
import copy

class test_replayMemory(unittest.TestCase):
    def setUp(self):
        self.test_replay = DQNAgent.ReplayMemory(5)
    
    def test_init_trivial(self):
        self.assertEqual = (5, self.test_replay.memory.maxlen)

    def test_init_lower_limit(self):
        with self.assertRaises(ValueError):
            test_replay = DQNAgent.ReplayMemory(-5)

    def test_push_wrong_tuple_size(self):
        with self.assertRaises(TypeError):
            self.test_replay.push(3)

    def test_push_change_memory(self):
        memory_old_size = len(self.test_replay.memory)
        state = torch.rand(2,3)
        next_state = torch.rand(2,3)
        action = torch.rand(2,3)
        reward = torch.rand(2,3)
        self.test_replay.push(state, action, next_state, reward)
        self.assertGreater(len(self.test_replay.memory), memory_old_size)
        self.assertEqual(self.test_replay.memory[-1], DQNAgent.Transition(state, action, next_state,reward))
    
    def test_sample(self):
        test_replay = DQNAgent.ReplayMemory(5)
        state = torch.rand(2,3)
        next_state = torch.rand(2,3)
        action = torch.rand(2,3)
        reward = torch.rand(2,3)
        for i in range(3):
            test_replay.push(state, action, next_state, reward)
        self.assertEqual(len(test_replay.sample(3)), 3)

    def test_sample_higher_limit(self):
        exceed_memory = len(self.test_replay.memory) + 1
        with self.assertRaises(ValueError):
            self.test_replay.sample(exceed_memory)

    def test_sample_lower_limit(self):
        with self.assertRaises(ValueError):
            self.test_replay.sample(-2)



class test_DQNAgent(unittest.TestCase):
    def setUp(self):
          self.agent = DQNAgent.DQNAgent(gym.make("LunarLander-v3",render_mode="human"), model_class=r"dqnModel_lunar_lander.pt")

    def test_agent_init(self):
        self.assertIsInstance(self.agent.policy_net, dqnModel.dqnModel)
        self.assertIsInstance(self.agent.target_net, dqnModel.dqnModel)

    def test_select_action_state_output(self):
        state =  torch.rand(2,3)
        output = self.agent.select_action(state)
        self.assertIsInstance(output, torch.Tensor)

    def test_select_action_state_not_tensor(self):
        with self.assertRaises(TypeError):
            self.agent.select_action(5)

    def test_soft_update(self):
        current_params = copy.deepcopy(self.agent.target_net.state_dict())
        self.agent.soft_update()
        new_params = dict(self.agent.target_net.state_dict())
        for k in self.agent.target_net.state_dict():
            self.assertFalse(torch.equal(new_params[k], current_params[k]))

    def test_train_param_not_int(self):
        with self.assertRaises(TypeError):
            self.agent.train('a')

if __name__ == '__main__':
    unittest.main()
