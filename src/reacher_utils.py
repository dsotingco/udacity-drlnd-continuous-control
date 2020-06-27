""" reacher_utils.py """

from unityagents import UnityEnvironment
import numpy as np

def collect_trajectories(env, policy_list):
    # initialize return variables
    prob_list = []
    state_list = []
    action_list = []
    reward_list = []

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

    # initialize actions matrix and probability matrix
    actions = np.zeros((num_agents, action_size))
    probs = np.zeros((num_agents, action_size))

    # run the agents in the environment
    while True:
        # actions = np.random.randn(num_agents, action_size)
        # actions = np.clip(actions, -1, 1)
        for agent_index in range(num_agents):
            agent_policy = policy_list[agent_index]
            agent_states = states[agent_index,:]
            (policy_actions, policy_log_probs) = agent_policy.forward(agent_states)
            actions[agent_index,:] = policy_actions.detach().numpy()
            probs[agent_index,:] = policy_log_probs.detach().numpy()
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = np.array(env_info.rewards)
        dones = env_info.local_done
        scores += env_info.rewards
        # Append results to output lists.
        assert isinstance(probs, np.ndarray)
        assert isinstance(states, np.ndarray)
        assert isinstance(actions, np.ndarray)
        assert isinstance(rewards, np.ndarray)
        prob_list.append(probs)
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)
        # Set up for next step
        states = next_states
        if np.any(dones):
            break

    average_agent_score = np.mean(scores)
    print('Total score (averaged over agents) this episode: {}'.format(average_agent_score))
    # print(actions)

    env.close()

    return prob_list, state_list, action_list, reward_list

def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1,
                      beta=0.01):
    pass
    # discount_array = discount **np.arange(len(rewards))
    # rewards = 