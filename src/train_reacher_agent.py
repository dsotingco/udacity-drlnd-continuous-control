"""train_reacher_agent.py
Train the Reacher agent.
"""

from unityagents import UnityEnvironment
import numpy as np
import torch
import reacher_utils
import ReacherPolicy

# Environment setup
env = UnityEnvironment(file_name="Reacher.exe")
num_agents = 20
policy_list = [ReacherPolicy.ReacherPolicy() for agent in range(num_agents)]

# Collect trajectories
(prob_list, state_list, action_list, reward_list) = reacher_utils.collect_trajectories(env, policy_list)
print(prob_list[0].shape)
print(state_list[0].shape)
print(action_list[0].shape)
print(reward_list[0].shape)

# Concatenate the trajectories (the agents are all the same, so just
# build up tensors for training)
prob_tensor = torch.cat([torch.tensor(prob_array) for prob_array in prob_list])            # N x 4
state_tensor = torch.cat([torch.tensor(state_array) for state_array in state_list])        # N x 33
action_tensor = torch.cat([torch.tensor(action_array) for action_array in action_list])    # N x 4

reward_array = reacher_utils.process_rewards(reward_list)
reward_tensor = torch.tensor(np.reshape(reward_array, -1))

print(prob_tensor.shape)
print(state_tensor.shape)
print(action_tensor.shape)
print(reward_tensor.shape)