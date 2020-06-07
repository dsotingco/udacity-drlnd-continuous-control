"""train_reacher_agent.py
Train the Reacher agent.
"""

from unityagents import UnityEnvironment
import numpy as np
import reacher_utils
import ReacherPolicy

env = UnityEnvironment(file_name="Reacher.exe")
num_agents = 20
policy_list = [ReacherPolicy.ReacherPolicy() for agent in range(num_agents)]

reacher_utils.collect_trajectories(env, policy_list)
