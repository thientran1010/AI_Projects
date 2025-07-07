
import yaml

class HParams:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.n_neurons = config['model'].get('n_neurons', 128)
        self.n_layers = config['model'].get('n_layers', 1)
        self.num_episodes = config['model'].get('num_episodes', 50)
        self.model_path = config['model'].get('model_path', 'new_dqnModel_lunar_lander.pt')

    def get_n_neurons(self):
        return self.n_neurons

    def get_n_layers(self):
        return self.n_layers

    def get_num_episodes(self):
        return self.num_episodes

    def get_model_path(self):
        return self.model_path
