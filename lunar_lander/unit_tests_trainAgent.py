import unittest
#import TrainAgent
#import HParams

class TrainAgent:
    def __init__(self, path, num_ep, overwrite, params):
        self.path = path
        self.num_episode = num_ep
        self.overwrite = overwrite
        self.hparams = params
    
class HParams:
    def __init__(self,n_neurons, n_layers, num_episodes):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.num_episodes = num_episodes

class test_TrainAgent(unittest.TestCase):
    
    def setUp(self):
        self.path = 'test path'
        self.num_ep = 3
        self.overwrite = False
        self.params = HParams(3,3,3)
        self.test_ta = TrainAgent(self.path, self.num_ep, self.overwrite, self.params)

    def test_init_parameters(self):
        self.assertIsInstance(str, self.test_ta.path)
        self.assertIsInstance(int, self.test_ta.num_episode)
        self.assertIsInstance(bool, self.test_ta.overwrite)
        self.assertIsInstance(HParams, self.test_ta.hparams)
    
    def test_init_path_not_str(self):
        with self.assertRaises(TypeError):
            test_ta = TrainAgent(5, self.num_ep, self.overwrite, self.params)

    def test_init_n_eps_not_int(self):
        with self.assertRaises(TypeError):
            test_ta = TrainAgent(self.path, 'a', self.overwrite, self.params)

    def test_init_overwrite_not_bool(self):
        with self.assertRaises(TypeError):
            test_ta = TrainAgent(self.path, self.num_ep, 5, self.params)

    def test_init_params_not_hparams(self):
        with self.assertRaises(TypeError):
            test_ta = TrainAgent(self.path, self.num_ep, self.overwrite, 5)

    def test_train_agent(self):
        pass

    def test_get_model_path(self):
        self.assertEqual(self.test_ta.get_model_path(), self.path)

    