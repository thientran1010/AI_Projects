import unittest
from hparams import HParams

class TestHParams(unittest.TestCase):
    #to compare actual vs aexpected values
    def test_default_values(self):
        hp = HParams()
        self.assertEqual(hp.get_n_neurons(), 128)
        self.assertEqual(hp.get_n_layers(), 1)
        self.assertEqual(hp.get_num_episodes(), 50)

    #to check if getter method returns properly
    def test_custom_initialization(self):
        hp = HParams(n_neurons=256, n_layers=3, num_episodes=100)
        self.assertEqual(hp.get_n_neurons(), 256)
        self.assertEqual(hp.get_n_layers(), 3)
        self.assertEqual(hp.get_num_episodes(), 100)

    #to test setter methods
    def test_setters(self):
        hp = HParams()
        hp.set_n_neurons(64)
        hp.set_n_layers(2)
        hp.set_num_episodes(200)

        self.assertEqual(hp.get_n_neurons(), 64)
        self.assertEqual(hp.get_n_layers(), 2)
        self.assertEqual(hp.get_num_episodes(), 200)

if __name__ == '__main__':
    unittest.main()
