
"""
   Deep Q-Network (DQN) model with configurable number of hidden layers and neurons.

   This neural network takes a state (observation) as input and outputs Q-values
   for each possible action. The architecture consists of:
     - An input layer
     - A configurable number of hidden layers
     - An output layer matching the number of actions

   First Author: TT
   
   Modifications by AF
   Date: 06-28-2025
      - layer2 attribute is changed to torch.nn.Sequential where a variable number of layers can be chosen.
      - New input variables: n_neurons and n_layers
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import HParams

class dqnModel(nn.Module):

    """
    Deep learning model with adjustable number of linear layers.

    Args:
        n_observations (int): Number of input features (state dimensions).
        n_actions (int): Number of output features (action dimensions).
        n_neurons (int, optional): Number of neurons in each hidden layer.
                                   If None, uses default from HParams.
        n_layers (int, optional): Number of hidden layers.
                                  If None, uses default from HParams.
    """

    def __init__(self, n_observations, n_actions, n_neurons = None, n_layers = None):

        super(dqnModel, self).__init__()

        hp = HParams()
        self.n_neurons = n_neurons if n_neurons is not None else hp.get_n_neurons()
        self.n_layers = n_layers if n_layers is not None else hp.get_n_layers()

        self.layer1 = nn.Linear(n_observations, self.n_neurons)

        #will add any number of layers. Default is set to 1
        self.layer2 = torch.nn.Sequential(*[nn.Linear(self.n_neurons, self.n_neurons) for i in range(self.n_layers)])

        self.layer3 = nn.Linear(self.n_neurons, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).


    def forward(self, x):
        """
             Forward pass of the network.
             Applies ReLU activation after each linear layer except the final output.
             Args:
                 x (torch.Tensor): Input tensor representing one or more observations.
             Returns:
                 torch.Tensor: Output Q-values for each action.
                               Shape: (batch_size, n_actions)
             """

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
