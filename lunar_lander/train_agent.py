"""
train_agent.py

Handles the training logic for a DQNAgent using the LunarLander-v3 environment.
Supports overwriting existing models and configurable training episodes and hyperparameters.

Author: AF
Created: 28-06-2025

Classes:
    - TrainAgent: Encapsulates training behavior, model saving, and hyperparameter management.
"""


from DQNAgent import DQNAgent
import gymnasium as gym
from hparams import HParams
import os 

class TrainAgent:
    """
       Manages the training process of a DQNAgent with support for model saving and overwriting.

       Attributes:
           model_path (str): File path where the trained model will be saved.
           num_episodes (int): Number of episodes to train the agent.
           overwrite (bool): Whether to overwrite an existing model file.
           hparams (HParams): Object containing hyperparameters for the model.
       """
    def __init__(
            self,
            model_path: str = "new_dqnModel_lunar_lander.pt",
            num_episodes: int = 50,
            overwrite: bool=False,
            hparams: HParams = None,
    ):
        """
            Initializes the TrainAgent instance.

            Args:
                    model_path (str, optional): Path to save the trained model. Defaults to 'new_dqnModel_lunar_lander.pt'.
                    num_episodes (int, optional): Number of episodes for training. Defaults to 50.
                    overwrite (bool, optional): If True, will delete and retrain existing model. Defaults to False.
                    hparams (HParams, optional): Optional hyperparameter object. Defaults to a new HParams instance.
                """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.overwrite = overwrite
        self.hparams = hparams or HParams()  # Default fallback

    def train_agent(self):
        """
            Trains a DQNAgent using the specified environment and hyperparameters.

            Checks if a model already exists:
                  - If `overwrite` is True, deletes the existing model and retrains.
                  - If `overwrite` is False, skips training to avoid overwriting.

            After training, the model is saved to the specified path.
                """
        if os.path.exists(self.model_path):
            if self.overwrite:
                print(f"Overwriting existing model at {self.model_path}...")
                os.remove(self.model_path)

            else:
                print(f"Model already exists at {self.model_path}. Skipping training.")
                return



        if self.num_episodes is None:
            self.num_episodes = 50

        env = gym.make("LunarLander-v3")
        agent = DQNAgent(env=env, hparams=self.hparams,model_class=self.model_path)
        agent.train(num_episodes=self.num_episodes)
        agent.save(self.model_path)
        print("Training complete and model saved.")

    def get_model_path(self):
        return self.model_path