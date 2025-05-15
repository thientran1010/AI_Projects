import os
os.chdir(r'C:\Users\MV\Downloads\SMBs\frozen_lake')
import numpy
from dqnModel import dqnModel
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)





env = gym.make("FrozenLake-v1",is_slippery=True, render_mode='human')
n_actions = env.action_space.n
n_observations = 16
target_net = dqnModel(n_observations, n_actions).to(device)

model_path=r'dqnModel.pt'
target_net.load_state_dict(torch.load(model_path, weights_only=True))



if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 50
else:
    num_episodes = 25

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state=[0 if state!=i else 1 for i in range(16)]
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    observation=state
    for t in count():
        
        action=(target_net(state).max(1).indices.view(1, 1)).item()
        observation, reward, terminated, truncated, _ = env.step(action)
        
        observation=[0 if observation!=i else 1 for i in range(16)]
        if terminated and reward==0:
            reward -=5
        elif reward ==0:
            reward -=1
        else:
            reward+=50
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        

        
        
        
        if done:
            
            break
env.close()
observation=3
[0 if observation!=i else 1 for i in range(16)]
observation
state
a = (target_net(state).max(1).indices.view(1, 1)).item()
a
state

target_net(state)
a.item()