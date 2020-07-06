import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReacherPolicy(nn.Module):
    """ Policy model. """

    def __init__(self, state_size=33, hidden1_size=128, hidden2_size=64, hidden3_size=32, action_size=4, 
                 init_std_deviation=1.0):
        super(ReacherPolicy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.fc4 = nn.Linear(hidden3_size, action_size)
        # Output of neural network: [mu1; mu2; mu3; mu4]
        self.std_deviations = nn.Parameter(init_std_deviation * torch.ones(1, self.action_size))

    def calculate_distribution_means(self, state):
        """ Calculate mean values to be used, using the neural network. """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        means = torch.tanh(x)
        return means

    def forward(self, state, use_sampling=True):
        """ Run the neural network and sample the distribution for actions. """
        state = torch.tensor(state, dtype=torch.float)
        self.train()
        means = self.calculate_distribution_means(state)
        m = torch.distributions.normal.Normal(means, self.std_deviations)
        if use_sampling:
            raw_nn_actions = m.sample()
        else:
            raw_nn_actions = means
        # Since distribution sampling may yield values beyond [-1, 1],
        # saturate the action values.
        actions = torch.clamp(raw_nn_actions, -1.0, 1.0)
        log_probs = m.log_prob(actions)
        return (actions, log_probs)

    def calculate_log_probs_from_actions(self, state, actions):
        """ Calculate log probabilities from state and actions.  
            To be used for calculating new probabilities as the 
            policy changes during training. """
        # NOTE: These are technically not log probabilities, but rather
        # logs of the probability density functions. 
        print("actions.shape: ", actions.shape)
        self.train()
        means = self.calculate_distribution_means(state)
        m = torch.distributions.normal.Normal(means, self.std_deviations)
        print("std deviations: ", self.std_deviations)
        log_probs = m.log_prob(actions)
        return log_probs
