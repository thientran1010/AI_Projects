import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from itertools import count
from dqnModel import dqnModel
from enum import Enum
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
from enum import Enum
import os
class DQNAGENT_MODE(Enum):
    EXPLORE="EXPLORE"
    EXPLOIT="EXPLOIT"

class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Store a transition in memory"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
class DQNAgent:
    

    def __init__(
        self,
        env: gym.Env,
        mode=DQNAGENT_MODE.EXPLORE,
        model_class: dqnModel=None,
        memory_capacity: int = 10000,
        batch_size: int = 128,
        gamma: float = 0.99,
        eps_start: float = 0.9,
        eps_end: float = 0.05,
        eps_decay: float = 1000,
        tau: float = 0.005,
        lr: float = 1e-4,
        device: torch.device = None,
        state_processor=None,
    ):
        # Environment and device setup
        self.env = env
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.n_actions = env.action_space.n

        # Initialize mode
        self.mode = mode
        # State preprocessor: maps raw observation to tensor
        self.state_processor = state_processor or (lambda s: torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0))

        # Determine number of observations from initial state
        init_state, _ = env.reset()
        proc_state = self.state_processor(init_state)
        self.n_observations = proc_state.shape[-1]
        
        # Networks
        try:
            self.policy_net = dqnModel(self.n_observations, self.n_actions).to(self.device)
            self.target_net = dqnModel(self.n_observations, self.n_actions).to(self.device)
            if  not os.path.exists(model_class):
                print("Path to model not found. Create a new model to:")
                print(model_class)
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                print("Loading existing model from: ")
                print(model_class)
                self.load(model_class)
        except Exception as e:
            print("Model not loaded correctly")
            print(f"An error occurred: {e}")
            import sys
            sys.exit()
        

        # Optimizer and hyperparameters
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.steps_done = 0
        self.episode_durations = []

        # Enable interactive plotting
        plt.ion()

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        if self.mode==DQNAGENT_MODE.EXPLORE:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            return self.policy_net(state).max(1).indices.view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Prepare batches
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute current Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next Q values
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute expected Q values
        expected_values = reward_batch + (self.gamma * next_state_values)

        # Optimize
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update(self):
        for key in self.policy_net.state_dict():
            self.target_net.state_dict()[key].copy_(
                self.policy_net.state_dict()[key] * self.tau
                + self.target_net.state_dict()[key] * (1 - self.tau)
            )

    def plot_durations(self, show_result: bool = False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        plt.clf()
        plt.title('Result' if show_result else 'Training...')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)

    def train(self, num_episodes: int):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.state_processor(state)
            total_reward = 0
            for t in count():
                action = self.select_action(state)
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated

                # Convert reward to tensor
                reward_tensor = torch.tensor([reward], device=self.device)

                # Prepare next state
                next_state = None if terminated else self.state_processor(obs)

                # Store transition
                self.memory.push(state, action, next_state, reward_tensor)

                # Move to next state
                state = next_state
                total_reward += reward

                # Perform optimization and update
                self.optimize_model()
                self.soft_update()

                if done:
                    self.episode_durations.append(total_reward)
                    self.plot_durations()
                    break
        plt.ioff()

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    



    def test_demonstration(self):
        
        self.mode=DQNAGENT_MODE.EXPLOIT
        observation, info = self.env.reset(seed=42)
        sumRe=0
        for step in range(10):
            observation, info = self.env.reset()
            state = self.state_processor(observation)
            
            cuRe=0
            terminated=False
            truncated=False
            while not (terminated or truncated):
                print("terminated ", terminated," truncated: ", truncated)
            
                # this is where you would insert your policy
                  
                action=self.policy_net(state).max(1).indices.view(1, 1).item()
                
                # step (transition) through the environment with the action
                # receiving the next observation, reward and if the episode has terminated or truncated
                observation, reward, terminated, truncated, info = self.env.step(action)
                next_state = None if terminated else self.state_processor(observation)
                
                
                
                state=next_state    
                cuRe+=reward
            sumRe+=cuRe
            avg_interval=10
            if step%avg_interval==0:
                print(sumRe/avg_interval)
                sumRe=0
                
            


        self.env.close()
        self.mode=DQNAGENT_MODE.EXPLORE


