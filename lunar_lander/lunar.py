import gymnasium as gym
import numpy as np
import random
import pickle
default_path=r"C:\Users\MV\Downloads\SMBs\lunar_lander"
import os
os.chdir(default_path)

model_path=r"lunar_model1.pkl"
reward_mem={}
with open(model_path, "rb") as file:
    reward_mem = pickle.load(file)

epsilon=1.1
def discrete(obs):
    state=[0]*len(obs)
    for i,v in enumerate(obs):
        state[i]=round(obs[i], 2)
    return tuple(state)
# Initialise the environment
#env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("LunarLander-v3")

# Reset the environment to generate the first observation

min_max=[[10000,-10000] for i in range(8)]
sumRe=0
truncated=0
for step in range(1000):
    observation, info = env.reset(seed=42)
    state=discrete(observation)
    cuRe=0
    terminated=False
    while not (terminated or truncated):
    #while not (terminated):
        # this is where you would insert your policy
        action = env.action_space.sample()
        val = random.uniform(0, 1)
        
        if val <= epsilon:
            
            # Exploration: randomly choose an action
            
            action = env.action_space.sample()
        else:
            
            if state not in reward_mem: action = env.action_space.sample()
            else:
                action= np.argmax(reward_mem[state])
                
        action= np.int64(action)
        
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)
        next_state = discrete(observation)
        if state not in reward_mem: reward_mem[state]=[0]*4
        if next_state not in reward_mem: reward_mem[next_state]=[0]*4
        action=int(action)
        reward_mem[state][action] = reward_mem[state][action] + 0.9 * (reward + 0.8 * np.max(reward_mem[next_state])-reward_mem[state][action])
        
        state=next_state    
        cuRe+=reward
    sumRe+=cuRe
    avg_interval=10
    if step%avg_interval==0:
        print(sumRe/avg_interval)
        sumRe=0
        with open(model_path, "wb") as file:
            pickle.dump(reward_mem, file) 
    if step%100:
        epsilon-=0.1


env.close()
with open(model_path, "wb") as file:
    pickle.dump(reward_mem, file)

print(model_path)



