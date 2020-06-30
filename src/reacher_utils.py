""" reacher_utils.py """

from unityagents import UnityEnvironment
import numpy as np
import torch

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
        # TODO: probably want to stack these so that I'm not looping over
        # 20 agents all the time downstream.  Maybe do in a separate function.

    average_agent_score = np.mean(scores)
    print('Total score (averaged over agents) this episode: {}'.format(average_agent_score))
    # print(actions)

    env.close()

    return prob_list, state_list, action_list, reward_list

# TODO: look at everything below to see if it would be better
# processed as tensors.  Rewards still need to be processed by episode
# in order to properly calculate future rewards (but after that it could
# be turned into a tensor.)
def process_rewards(reward_list, discount=0.995):
    """ Process the rewards for one run of collect_trajectories().  
    Outputs normalized, discounted, future rewards as a matrix of 
    num_agents rows, and num_timesteps columns."""
    # calculate discounted rewards
    num_timesteps = len(reward_list)
    num_agents = len(reward_list[0])
    reward_matrix = np.asarray(reward_list).T    # rows are agents; columns are time
    discount_array = discount**np.arange(len(reward_list))
    discount_matrix = np.tile(discount_array, (num_agents, 1))
    discounted_rewards = reward_matrix * discount_matrix

    # calculate future discounted rewards
    future_rewards = np.fliplr( np.fliplr(discounted_rewards).cumsum(axis=1) )

    # normalize the future discounted rewards
    mean = np.mean(future_rewards, axis=1)
    std = np.std(future_rewards, axis=1) + 1.0e-10
    mean_matrix = np.tile(mean[np.newaxis].T, (1,num_timesteps))
    std_matrix  = np.tile(std[np.newaxis].T, (1,num_timesteps))
    normalized_rewards = (future_rewards - mean_matrix) / std_matrix
    stacked_normalized_rewards = np.reshape(normalized_rewards, -1)
    return stacked_normalized_rewards

def calculate_new_log_probs(policy, state_list, action_list):
    """ Calculate new log probabilities of the actions, 
        given the states.  To be used during training as the
        policy is changed by the optimizer. """
    new_prob_list = [policy.calculate_log_probs_from_actions(s, a) for s, a in zip(state_list, action_list)]
    return new_prob_list

def clipped_surrogate(policy, old_prob_list, state_list, action_list, reward_list,
                      discount=0.995,
                      epsilon=0.1,
                      beta=0.01):
    normalized_rewards = process_rewards(reward_list, discount)
    new_prob_list = calculate_new_log_probs(policy, state_list, action_list)
    # TODO: calculate probability ratio; clipped function; regularization/entropy term