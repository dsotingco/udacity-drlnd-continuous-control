"""train_reacher_agent.py
Train the Reacher agent.
"""

from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.optim as optim
import reacher_utils
import ReacherPolicy

# Hyperparameters
learning_rate = 1e-4
num_episodes = 1

# Environment setup
env = UnityEnvironment(file_name="Reacher.exe")
num_agents = 20
policy = ReacherPolicy.ReacherPolicy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Initialize scores, etc.
episode_scores = []

# Collect trajectories
for episode in range(num_episodes):
    (prob_list, state_list, action_list, reward_list) = reacher_utils.collect_trajectories(env, policy)

print(prob_list[0].shape)
print(state_list[0].shape)
print(action_list[0].shape)
print(reward_list[0].shape)

# Concatenate the trajectories (the agents are all the same, so just
# build up tensors for training)
old_prob_nparray = np.concatenate(prob_list, axis=0)
state_nparray = np.concatenate(state_list, axis=0)
action_nparray = np.concatenate(action_list, axis=0)
reward_nparray = reacher_utils.process_rewards(reward_list)

print(old_prob_nparray.shape)
print(state_nparray.shape)
print(action_nparray.shape)
print(reward_nparray.shape)

new_prob_batch = reacher_utils.calculate_new_log_probs(policy, torch.tensor(state_nparray), torch.tensor(action_nparray))
print(new_prob_batch.shape)