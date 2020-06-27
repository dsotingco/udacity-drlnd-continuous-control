import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReacherPolicy(nn.Module):
    """ Policy model. """

    def __init__(self, state_size=33, hidden1_size=128, hidden2_size=64, action_size=4, 
                 init_std_deviation=1.0):
        super(ReacherPolicy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 2*action_size)
        # Output of neural network: [mu1; mu2; mu3; mu4; sigma1; sigma2; sigma3; sigma4]
        self.means = torch.tensor([0.0] * self.action_size)
        self.std_deviations = torch.tensor([init_std_deviation] * self.action_size)

    def calculate_distribution_params(self, state):
        """ Calculate means and standard deviations to be used. """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = torch.tanh(x)
        self.means = out.flatten()[0:4]
        self.std_deviations = out.flatten()[4:]

    def forward(self, state, use_sampling=True):
        """ Run the neural network and sample the distribution for actions. """
        actions = torch.tensor([0] * self.action_size)
        log_probs = torch.tensor([0] * self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.eval()
        self.calculate_distribution_params(state)
        m = torch.distributions.normal.Normal(self.means, self.std_deviations)
        if use_sampling:
            actions = m.sample()
        else:
            actions = self.means
        log_probs = m.log_prob(actions)
        return (actions, log_probs)

    def calculate_log_probs_from_actions(self, state, actions):
        """ Calculate log probabilities from state and actions.  
            To be used for calculating new probabilities as the 
            policy changes during training. """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.eval()
        self.calculate_distribution_params(state)
        m = torch.distributions.normal.Normal(self.means, self.std_deviations)
        log_probs = m.log_prob(actions)
        return log_probs
