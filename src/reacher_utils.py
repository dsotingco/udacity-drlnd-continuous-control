""" reacher_utils.py """

from unityagents import UnityEnvironment
import numpy as np

def collect_trajectories(env, policy):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    # get initial states and scores for each agent
    states = env_info.vector_observations
    scores = np.zeros(num_agents)

    # run the agents in the environment
    # TODO: use the neural network
    while True:
        actions = np.random.randn(num_agents, action_size)
        actions = np.clip(actions, -1, 1)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break

    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    env.close()

    # TODO: return old_probs, states, actions, rewards
