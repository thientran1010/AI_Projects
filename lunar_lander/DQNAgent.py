"""
    Deep Q-Network (DQN) Agent for training and testing in reinforcement learning environments.


    First author: TT
    
    Modifications and Extensions by AF
    Date: 28-06-2025

    Attribute modifications:
        -	added attribute hparams, an HParams object to use hyperpapramers

    dqnModel calls take two additional parameters: num_neurons and num_layers

    Modifications to existing functions:
        -	train: Added logic to collect losses during training and output graph at the end
        -	test_demonstration: introduced logic to output the reward progress graph and output the last reward and the average of the last three rewards
    New function
        -   plot_loss_curve(): plots the loss curve during training with a moving average emphasis.

"""


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
import numpy as np

# import params
from hparams import HParams


class DQNAGENT_MODE(Enum):

    EXPLORE="EXPLORE"
    EXPLOIT="EXPLOIT"

class ReplayMemory:
    """
        A cyclic buffer that holds the agent's experience tuples (transitions)
        for experience replay during training.

        Attributes:
            memory (deque): A bounded deque that stores Transition tuples.
        """

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Store a transition in memory"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        """Randomly samples a batch of transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current number of stored transitions."""
        return len(self.memory) # Fix: return length of the deque

class DQNAgent:

    """
        Deep Q-Network (DQN) Agent for solving reinforcement learning problems using the DQN algorithm.

        This agent is designed for environments with discrete action spaces and supports training,
        testing, and evaluation with experience replay and soft target updates.

        Args:
            env (gym.Env): The OpenAI Gym environment.
            mode (Enum): Mode of operation (EXPLORE or EXPLOIT).
            model_class (str): Path to the saved model file. Defaults to None.
            memory_capacity (int): Capacity of the experience replay memory.
            batch_size (int): Batch size for sampling from memory.
            gamma (float): Discount factor.
            eps_start (float): Starting value of epsilon for epsilon-greedy policy.
            eps_end (float): Final value of epsilon.
            eps_decay (float): Controls decay rate of epsilon.
            tau (float): Soft update parameter.
            lr (float): Learning rate.
            device (torch.device): PyTorch device (CPU or CUDA).
            state_processor (callable): Function to process raw observations.
            hparams (HParams): Object containing model architecture hyperparameters.

        """

    def __init__(
        self,
        env: gym.Env,
        mode=DQNAGENT_MODE.EXPLORE,
        model_class: str = None, # Changed default to None initially
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
        hparams = None
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

        #hparams:
        self.hparams = hparams
        self.n_neurons = self.hparams.get_n_neurons() # <--- hparams number of neurons
        self.n_layers = self.hparams.get_n_layers()   # <---- hparams number of layers


        # Default model path if not provided
        if model_class is None:
            self.model_path = f"dqnModel_{env.spec.id}.pt" # Construct default path
        else:
            self.model_path = model_class


        # Networks
        try:
            self.policy_net = dqnModel(self.n_observations, self.n_actions, self.n_neurons, self.n_layers).to(self.device)
            self.target_net = dqnModel(self.n_observations, self.n_actions, self.n_neurons, self.n_layers).to(self.device)
            if  not os.path.exists(self.model_path): # Use self.model_path here
                print("Path to model not found. Creating a new model at:")
                print(self.model_path)
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                print("Loading existing model from: ")
                print(self.model_path)
                self.load(self.model_path) # Use self.model_path here
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
        """
                Selects an action using an epsilon-greedy strategy during exploration.
                Args:
                    state (torch.Tensor): Current environment state.
                Returns:
                    torch.Tensor: Chosen action.
                """
        if self.mode==DQNAGENT_MODE.EXPLORE:
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
            steps_done = self.steps_done + 1
            self.steps_done = steps_done
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            return self.policy_net(state).max(1).indices.view(1, 1)


    def optimize_model(self):
        """
            Samples a batch from memory and performs a single optimization step on the policy network.
            Returns:
                float: The loss value for the optimization step (if available).
                """
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

        # new line:
        return loss.item() # -------------------return loss

    def soft_update(self):
        """
            Performs a soft update of the target network parameters using the policy network.
                """
        for key in self.policy_net.state_dict():
            self.target_net.state_dict()[key].copy_(
                self.policy_net.state_dict()[key] * self.tau
                + self.target_net.state_dict()[key] * (1 - self.tau)
            )

    plt.ion()

    def plot_durations(self, show_result: bool = False):
        """
            Plots the score (total reward) over episodes.

            Args:
                show_result (bool): If True, labels the plot as final result.
               """
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

    # loss curve plot function <---- new
    def plot_loss_curve(self, loss_history):
        """
            Plots the training loss curve and a moving average smoothed version.
            Args:
                loss_history (list): List of loss values recorded during training.
        """
        plt.figure(2)
        plt.title('Training Loss Over Time')
        plt.xlabel('Training Step')
        plt.ylabel('Loss (Smooth L1)')

        if len(loss_history) > 0:
            plt.plot(loss_history, alpha=0.6, label='Raw Loss')

            # Plot smoothed loss (moving average)
            if len(loss_history) >= 100:
                window_size = 100
                smoothed_loss = []
                for i in range(window_size, len(loss_history)):
                    smoothed_loss.append(np.mean(loss_history[i-window_size:i]))
                plt.plot(range(window_size, len(loss_history)), smoothed_loss,
                        'r-', linewidth=2, label=f'{window_size}-step Moving Average')

            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.show()


    def train(self, num_episodes: int):
        """
            Trains the agent using the DQN algorithm over a specified number of episodes.
            Args:
                num_episodes (int): Number of episodes to train the agent.
               """

        loss_history = []            # < --------------------- collects loss history
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
                loss = self.optimize_model()  # <--------- return loss to collect for plot

                # collect loss
                if loss is not None:
                  loss_history.append(loss)

                self.soft_update()


                if done:
                    self.episode_durations.append(total_reward)
                    self.plot_durations()
                    break

        plt.ioff()

        # added lines to show the final plots of reward and loss curves
        self.plot_durations(show_result=True)
        self.plot_loss_curve(loss_history)
        plt.show()



    def save(self, path: str):
        """
            Saves the current policy network to a file.
            Args:
                path (str): Path to save the model file.
               """
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        """
            Loads a policy network from a saved model file and syncs the target network.
            Args:
                path (str): Path to the saved model file.
                """
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def test_demonstration(self):
        """
            Runs a demonstration of the trained agent in the environment.
            Switches to exploit mode and displays the agent's performance visually.
            Shows reward graphs at the end of testing.
            """

        self.mode=DQNAGENT_MODE.EXPLOIT
        observation, info = self.env.reset(seed=42)
        sumRe=0
        num_steps = 10 #<--- cleaner that way than hardcoded below
        for step in range(num_steps):
            observation, info = self.env.reset()
            state = self.state_processor(observation)

            cuRe=0
            terminated=False
            truncated=False
            while not (terminated or truncated):
                # print("terminated ", terminated," truncated: ", truncated)

                # this is where you would insert your policy

                action=self.policy_net(state).max(1).indices.view(1, 1).item()

                # step (transition) through the environment with the action
                # receiving the next observation, reward and if the episode has terminated or truncated
                observation, reward, terminated, truncated, info = self.env.step(action)
                next_state = None if terminated else self.state_processor(observation)



                state=next_state
                cuRe+=reward

            self.episode_durations.append(cuRe) # < --- log reward to show reward graph at the end
            sumRe+=cuRe


            avg_interval=10
            if step%avg_interval==0:
                print(sumRe/avg_interval)
                sumRe=0
            if step == num_steps -1:
              self.plot_durations(show_result=True)
              plt.ioff()
              plt.tight_layout()
              plt.show()
              input("Press Enter to continue...")


        self.env.close()
        self.mode=DQNAGENT_MODE.EXPLORE

        # logic to print the average reward. Alternatively cound print all rewards in a list or both
        num_last_steps = 3
        print(f"Last episode reward: {self.episode_durations[-1]:.2f}")
        if len(self.episode_durations) >= num_last_steps:
            avg_reward = sum(self.episode_durations[-num_last_steps:]) / num_last_steps
            print(f"Average reward over last {num_last_steps} episodes: {avg_reward:.2f}")
            print(f"The lander landed after {num_steps} episodes.")

