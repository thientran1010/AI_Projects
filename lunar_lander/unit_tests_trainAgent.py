import unittest
import train_agent
import hparams

class test_TrainAgent(unittest.TestCase):
    
    def setUp(self):
        self.path = 'test_model.pt'
        self.num_ep = 3
        self.overwrite = False
        self.params = hparams.HParams(3,3,3)
        self.test_ta = train_agent.TrainAgent(self.path, self.num_ep, self.overwrite, self.params)

    def test_init_parameters(self):
        self.assertIsInstance(self.test_ta.get_model_path(), str)
        self.assertIsInstance(self.test_ta.num_episodes, int)
        self.assertIsInstance(self.test_ta.overwrite, bool)
        self.assertIsInstance(self.test_ta.hparams, hparams.HParams)
    
    def test_init_path_not_str(self):
        with self.assertRaises(TypeError):
            test_ta = train_agent.TrainAgent(5, self.num_ep, self.overwrite, self.params)

    def test_init_n_eps_not_int(self):
        with self.assertRaises(TypeError):
            test_ta = train_agent.TrainAgent(self.path, 'a', self.overwrite, self.params)

    def test_init_overwrite_not_bool(self):
        with self.assertRaises(TypeError):
            test_ta = train_agent.TrainAgent(self.path, self.num_ep, 5, self.params)

    def test_init_params_not_hparams(self):
        with self.assertRaises(TypeError):
            test_ta = train_agent.TrainAgent(self.path, self.num_ep, self.overwrite, 5)

    def test_train_agent(self):
        pass

    def test_get_model_path(self):
        self.assertEqual(self.test_ta.get_model_path(), self.path)

    
if __name__ == '__main__':
    unittest.main()