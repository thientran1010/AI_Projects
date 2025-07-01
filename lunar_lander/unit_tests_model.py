import unittest
import torch as torch
import dqnModel as model

class test_model(unittest.TestCase):

    def setUp(self):
        #Both the parameters and the model are put as gobal variables here for ease of modification
        self.n_observations = 20
        self.n_actions = 30
        self.test_model =  model.dqnModel(self.n_observations,self.n_actions)

    #The next three functions all verify if the layers set in the init of the model have the expected form
    def test_init_model_trivial_layer1(self):
        self.assertEqual(self.n_observations, self.test_model.layer1.in_features)
        self.assertEqual(128, self.test_model.layer1.out_features)

    def test_init_model_trivial_layer2(self):
        self.assertEqual(128, self.test_model.layer2.in_features)
        self.assertEqual(128, self.test_model.layer2.out_features)
    
    def test_init_model_trivial_layer3(self):
        self.assertEqual(128, self.test_model.layer3.in_features)
        self.assertEqual(self.n_actions, self.test_model.layer3.out_features)
    
    #These next two tests verify inputting negative parameters send the right error. 
    #There is no higher limit on the parameters, so no need to check that.
    def test_init_lower_limit_action(self):
        with self.assertRaises(ValueError):
            test_model = model.dqnModel(10,-2)
    
    def test_init_lower_limit_observation(self):
        with self.assertRaises(ValueError):
            test_model = model.dqnModel(-2,10)

    #This test verifies that inputting non-integers will send the right error.
    def test_init_param_not_integer(self):
        with self.assertRaises(TypeError):
            test_model = model.dqnModel('a', 'b')

    #This test checks if the output of the forward functions returns a tensor of the right shape for the input
    def test_forward_output(self):
        input = torch.rand(1,self.n_observations)
        output = self.test_model.forward(input)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1,self.n_actions))

    #This test checks if the right error is called when a parameter that is not a tensor is given
    def test_forward_x_not_tensor(self):
        with self.assertRaises(TypeError):
            self.test_model.forward(5)


if __name__ == '__main__':
    unittest.main()
