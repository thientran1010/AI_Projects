import gym
#import gymnasium as gym

import random
import numpy as np
import pickle

import keyboard


env = gym.make("Taxi-v3").env

done=False
quit=False
eps=5000
epsilon=1.1
state_space=env.observation_space.n
action_space=env.action_space.n
# with open("reward_mem.pkl", "rb") as file:
#     reward_mem = pickle.load(file)
reward_mem = np.zeros((state_space, action_space))



def calRe():
    for i,(state,action,reward) in enumerate(seq[:-1]):
        next_state=seq[i+1][0]
        reward_mem[state,action]=reward_mem[state,action]+0.9*(reward+0.8*max(reward_mem[next_state,:])-reward_mem[state,action])


maxRe=-1000000
ep=0
while ep < eps and not quit:
    seq=[]
    cuRe=0
    state = env.reset()
    #env.render()
    done=False
    steps=0
    while not done and steps <100:
        val = random.uniform(0, 1)
        if val <= epsilon:
        
            # Exploration: randomly choose an action
            action = env.action_space.sample()
        else:
            
            action= np.argmax(reward_mem[state,:])
        #env.render()
        # Apply the action and see what happens
        next_state, reward, done, info = env.step(action)
        seq.append([state,action,reward])
        
        state = next_state
        cuRe+=reward
        steps+=1
        
        
        
        if keyboard.is_pressed('s'):
            print("Press stop, saving params")
            quit==True
            ep=50000




    if ep%500==0: 
        print("cure: ",cuRe,"|| max: ",maxRe)
        print("epsilon: ",epsilon)
    ep+=1

    maxRe=max(cuRe,maxRe)
    
    calRe()
    epsilon = np.exp(-0.0005*ep)
    # if ep%350==0:
    #     epsilon-=0.1






env.close()


with open("reward_mem.pkl", "wb") as file:
    pickle.dump(reward_mem, file)










# env = gym.make("Taxi-v3").env

# done=False
# eps=range(500)
# epsilon=1.1


# state = env.reset()
# env.render()
# maxRe=0
# import keyboard
# quit=False

# while not quit:
#     seq=[]
#     cuRe=0
#     while not done:
#         action= np.argmax(reward_mem[state])
#         env.render()
#         # Apply the action and see what happens
#         next_state, reward, done, info = env.step(action)
#         print(action)
#         #seq.append([state,action,reward])
#         state = next_state
#         cuRe+=reward
#         if keyboard.is_pressed('s'):
#             quit==True
#             done==True

#     maxRe=max(cuRe,maxRe)
#     print(maxRe)
#     #calRe()


# reward_mem[state]
# np.argmax(reward_mem[state])


# env.close()
