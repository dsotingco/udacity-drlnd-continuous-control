import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReacherPolicy(nn.Module):
    """ Policy model. """

    def __init__(self, state_size=33, hidden1_size=128, hidden2_size=64, action_size=4):
        super(ReacherPolicy, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.batchnorm = nn.BatchNorm1d(num_features=state_size)
        self.fc1 = nn.Linear(state_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, action_size)

    def forward(self, state):
        x = self.batchnorm(state)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = F.tanh(x)
        return out

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
