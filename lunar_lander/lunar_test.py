import gymnasium as gym
import numpy as np
import random
import pickle
default_path=r"C:\Users\MV\Downloads\SMBs\lunar_lander"
import os
os.chdir(default_path)
from dqnModel import dqnModel
import torch

model_path=r"dqnModel_lunar_lander.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")
#env = gym.make("LunarLander-v3")

# Reset the environment to generate the first observation

min_max=[[10000,-10000] for i in range(8)]
sumRe=0
truncated=0
epsilon=0
observation, info = env.reset(seed=42)
n_actions = env.action_space.n
n_observations = len(observation)

target_net = dqnModel(n_observations, n_actions).to(device)
target_net.load_state_dict(torch.load(model_path, weights_only=True))

for step in range(50):
    observation, info = env.reset()
    
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    cuRe=0
    terminated=False
    truncated=False
    while not (terminated or truncated):
        print("terminated ", terminated," truncated: ", truncated)
    
        # this is where you would insert your policy
        
        val = random.uniform(0, 1)
        
        if val <= epsilon:
            
            # Exploration: randomly choose an action
            
            action = env.action_space.sample()
        else:
            
            
            action= target_net(state).max(1).indices.view(1, 1).item()
                
        
        
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        
        
        state=next_state    
        cuRe+=reward
    sumRe+=cuRe
    avg_interval=10
    if step%avg_interval==0:
        print(sumRe/avg_interval)
        sumRe=0
        
    if step%100:
        epsilon-=0.1


env.close()

print(model_path)



