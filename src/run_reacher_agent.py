"""run_reacher_agent.py
Run the Reacher agent.
"""

from unityagents import UnityEnvironment
import numpy as np
import torch
import reacher_utils
import ReacherPolicy
import matplotlib.pyplot as plt
from collections import deque

n_episodes = 100

env = UnityEnvironment(file_name="Reacher.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
num_agents = 20

policy = ReacherPolicy.ReacherPolicy() 
policy.load_state_dict(torch.load('reacher_weights_solved_510.pth'))

scores_window = deque(maxlen=100)

for i_episode in range(0, n_episodes):
    scores = np.zeros(num_agents, dtype=np.float32)
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations.astype(np.float32)

    while True:
        (actions, _probs) = policy(torch.tensor(states, dtype=torch.float))
        env_info = env.step(actions.detach().numpy())[brain_name]
        next_states = env_info.vector_observations.astype(np.float32)
        rewards = np.array(env_info.rewards)
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break

    average_agent_score = np.mean(scores)
    #print('Total score (averaged over agents) this episode: {}'.format(average_agent_score))
    scores_window.append(average_agent_score)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_window)), scores_window)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

