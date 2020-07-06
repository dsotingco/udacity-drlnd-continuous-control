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
from collections import deque

# Hyperparameters
learning_rate = 2e-4
num_episodes = 3
discount = 0.995
epsilon = 0.1
beta = 0.01
batch_size = 128
num_epochs_per_episode = 5    # could probably go up to 15 with batch_size = 64

# Environment setup
env = UnityEnvironment(file_name="Reacher.exe")
num_agents = 20
policy = ReacherPolicy.ReacherPolicy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Initialize scores, etc.
episode_scores = []
scores_window = deque(maxlen=100)
score_threshold = 35

# Run episodes and train agent.
for episode in range(num_episodes):
    # Collect trajectories
    (prob_list, state_list, action_list, reward_list, average_agent_score) = reacher_utils.collect_trajectories(env, policy)

    # Process scores
    episode_scores.append(average_agent_score)
    scores_window.append(average_agent_score)
    if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
    if np.mean(scores_window) >= score_threshold:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
        torch.save(policy.state_dict(), 'checkpoint.pth')
        break

    # Run training epochs
    for epoch in range(num_epochs_per_episode):
        reacher_utils.run_training_epoch(policy, optimizer, prob_list, state_list, action_list, reward_list,
                                        discount=discount, epsilon=epsilon, beta=beta, batch_size=batch_size)

    # the clipping parameter reduces as time goes on
    epsilon*=.999
    
    # the regulation term also reduces
    # this reduces exploration in later runs
    beta*=.995

env.close()

# Plot scores
fig = plt.figure()
plt.plot(np.arange(len(episode_scores)), episode_scores)
plt.show()
