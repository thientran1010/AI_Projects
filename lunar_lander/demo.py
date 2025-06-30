import gymnasium as gym
import DQNAgent as DQNAgent
from hparams import HParams
"""
A quick demonstration to be run as a stand alone.
Set the default neurons and number of layers before executing.
"""
hp = HParams()
env = gym.make("LunarLander-v3",render_mode="human")
agent=DQNAgent.DQNAgent(env=env,model_class=r"dqnModel_lunar_lander.pt", hparams=hp)
agent.test_demonstration()


