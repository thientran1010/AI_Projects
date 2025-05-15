import gym
#import gymnasium as gym

import random
import numpy as np
import pickle

import keyboard


env = gym.make("Taxi-v3").env

done=False
quit=False
eps=1000
epsilon=1.1
state_space=env.observation_space.n
action_space=env.action_space.n
with open("reward_mem.pkl", "rb") as file:
    reward_mem = pickle.load(file)

# with open("q_table.pkl", "rb") as file:
#     reward_mem = pickle.load(file)



state = env.reset()
env.render()
maxRe=0
import keyboard
quit=False

while not quit:
    seq=[]
    cuRe=0
    state=env.reset()
    done=False
    while not done:
        action= np.argmax(reward_mem[state,:])
        env.render()
        # Apply the action and see what happens
        next_state, reward, done, info = env.step(action)
        #print(action)
        #seq.append([state,action,reward])
        state = next_state
        cuRe+=reward
        if keyboard.is_pressed('s'):
            quit==True
            done==True
    
    maxRe=max(cuRe,maxRe)
    print("cure: ",cuRe,"|| max: ",maxRe)
    print(maxRe)
    #calRe()




env.close()
