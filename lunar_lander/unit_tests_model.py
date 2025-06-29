import unittest
import torch as torch
import dqnModel as model

class test_model(unittest.TestCase):

    def setUp(self):
        self.n_observations = 20
        self.n_actions = 30
        self.test_model =  model.dqnModel(self.n_observations,self.n_actions)

    def test_init_model_trivial_layer1(self):
        self.assertEqual(self.n_observations, self.test_model.layer1.in_features)
        self.assertEqual(128, self.test_model.layer1.out_features)

    def test_init_model_trivial_layer2(self):
        self.assertEqual(128, self.test_model.layer2.in_features)
        self.assertEqual(128, self.test_model.layer2.out_features)
    
    def test_init_model_trivial_layer3(self):
        self.assertEqual(128, self.test_model.layer3.in_features)
        self.assertEqual(self.n_actions, self.test_model.layer3.out_features)
    
    def test_init_lower_limit_action(self):
        with self.assertRaises(ValueError):
            test_model = model.dqnModel(10,-2)
    
    def test_init_lower_limit_observation(self):
        with self.assertRaises(ValueError):
            test_model = model.dqnModel(-2,10)

    def test_forward_output(self):
        input = torch.rand(1,self.n_observations)
        output = self.test_model.forward(input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1,self.n_actions))

    def test_forward_x_not_tensor(self):
        with self.assertRaises(TypeError):
            self.test_model.forward(5)


if __name__ == '__main__':
    unittest.main()
