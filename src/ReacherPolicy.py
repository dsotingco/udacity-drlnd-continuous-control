import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReacherPolicy(nn.Module):
    """ Policy model. """

    def __init__(self, state_size=33, hidden1_size=128, hidden2_size=64, action_size=4):
        super(ReacherPolicy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        # self.batchnorm = nn.BatchNorm1d(num_features=state_size)
        self.fc1 = nn.Linear(state_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 2*action_size)
        # Output of neural network: [mu1; mu2; mu3; mu4; sigma1; sigma2; sigma3; sigma4]

    def forward(self, state):
        # x = self.batchnorm(state)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = torch.tanh(x)
        return out

    def act(self, state, use_sampling=True):
        actions = np.array([0] * self.action_size)
        log_probs = np.array([0] * self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.eval()
        distribution_params = self.forward(state).cpu()
        means = distribution_params.flatten()[0:4].detach().numpy()
        min_std_deviations = [0.1, 0.1, 0.1, 0.1]    # to avoid PyTorch numerical errors
        nn_std_deviations = distribution_params.flatten()[4:].detach().numpy()
        std_deviations = np.maximum(min_std_deviations, nn_std_deviations)
        mean_matrix = torch.Tensor(means)
        std_deviation_matrix = torch.Tensor(np.diagflat(std_deviations))
        m = MultivariateNormal(mean_matrix, std_deviation_matrix)
        if use_sampling:
            actions = m.sample().detach().numpy()
        else:
            actions = means
        # log_probs = m.log_prob(actions).detach().numpy()
        return (actions, log_probs)
