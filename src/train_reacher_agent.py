"""train_reacher_agent.py
Train the Reacher agent.
"""

from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.optim as optim
import reacher_utils
import ReacherPolicy
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 1e-4
num_episodes = 1
discount = 0.995
epsilon = 0.1
beta = 0.01
batch_size = 64

# Environment setup
env = UnityEnvironment(file_name="Reacher.exe")
num_agents = 20
policy = ReacherPolicy.ReacherPolicy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Initialize scores, etc.
episode_scores = []

# Run episodes and train agent.
for episode in range(num_episodes):
    (prob_list, state_list, action_list, reward_list, average_agent_score) = reacher_utils.collect_trajectories(env, policy)
    episode_scores.append(average_agent_score)
    reacher_utils.run_training_epoch(policy, optimizer, prob_list, state_list, action_list, reward_list,
                                     discount=discount, epsilon=epsilon, beta=beta, batch_size=batch_size)
    # TODO: can run multiple epochs per episode
    # TODO: running average of scores

env.close()

# Plot scores
fig = plt.figure()
plt.plot(np.arange(len(episode_scores)), episode_scores)
plt.show()

