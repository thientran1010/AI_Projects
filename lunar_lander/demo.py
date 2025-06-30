import os
#os.chdir(r'C:\Users\MV\Downloads\SMBs\lunar_lander')
import numpy
from dqnModel import dqnModel
import gymnasium as gym

import DQNAgent as DQNAgent





env = gym.make("LunarLander-v3",render_mode="human")
agent=DQNAgent.DQNAgent(env=env,model_class=r"dqnModel_lunar_lander.pt")
agent.test_demonstration()
