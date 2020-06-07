"""train_reacher_agent.py
Train the Reacher agent.
"""

from unityagents import UnityEnvironment
import numpy as np
import reacher_utils
import ReacherPolicy

env = UnityEnvironment(file_name="Reacher.exe")
policy = ReacherPolicy.ReacherPolicy()

reacher_utils.collect_trajectories(env, policy)
