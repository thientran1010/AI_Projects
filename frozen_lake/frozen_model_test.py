import gymnasium as gym
import numpy as np
import random
import pickle
default_path=r"C:\Users\MV\Downloads\SMBs\frozen_lake"
import os
os.chdir(default_path)

model_path=r"frozen_model1.pkl"
#model_path=r"frozen_model_no_slip.pkl"
reward_mem={}
with open(model_path, "rb") as file:
    reward_mem = pickle.load(file)
reward_test=reward_mem.copy()
epsilon=0

env = gym.make("FrozenLake-v1",is_slippery=True,render_mode='human')
#env = gym.make("FrozenLake-v1",is_slippery=False,render_mode='human')
#env = gym.make("FrozenLake-v1",is_slippery=True)


sumRe=0
truncated=0
for step in range(20):
    observation, info = env.reset()
    state=observation
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
            
            if state not in reward_test: action = env.action_space.sample()
            else:
                action= np.argmax(reward_test[state])
                
        action= np.int64(action)
        
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated and reward==0: 
            reward-=5
        elif reward==0:
            reward-=1
        else: reward+=50
        next_state = observation
        if state not in reward_mem: reward_mem[state]=[0]*4
        if next_state not in reward_mem: reward_mem[next_state]=[0]*4
        action=int(action)
        #reward_mem[state][action] = reward_mem[state][action] + 0.9 * (reward + 0.8 * np.max(reward_mem[next_state])-reward_mem[state][action])
        
        state=next_state    
        cuRe+=reward
        #print(reward)
    sumRe+=cuRe
    avg_interval=1
    if step%avg_interval==0:
        print(sumRe/avg_interval)
        sumRe=0
        
    if step%1000:
        epsilon-=0.1


env.close()

