""" reacher_utils.py """

from unityagents import UnityEnvironment
import numpy as np
import torch

def collect_trajectories(env, policy):
    """ TODO: document the outputs """
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
    states = env_info.vector_observations.astype(np.float32)
    scores = np.zeros(num_agents, dtype=np.float32)

    # run the agents in the environment
    while True:
        (policy_actions, policy_log_probs) = policy.forward(states)
        actions = policy_actions.detach().numpy()
        assert isinstance(actions, np.ndarray)
        probs = policy_log_probs.detach().numpy()
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations.astype(np.float32)
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

    return prob_list, state_list, action_list, reward_list, average_agent_score

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
    mean = np.mean(future_rewards, axis=0)
    std = np.std(future_rewards, axis=0) + 1.0e-10
    mean_matrix = np.tile(mean[np.newaxis], (num_agents,1))
    std_matrix  = np.tile(std[np.newaxis], (num_agents,1))
    normalized_rewards = (future_rewards - mean_matrix) / std_matrix
    stacked_normalized_rewards = np.reshape(normalized_rewards.T, -1)
    return stacked_normalized_rewards

def calculate_new_log_probs(policy, state_batch, action_batch):
    """ Calculate new log probabilities of the actions, 
        given the states.  To be used during training as the
        policy is changed by the optimizer. 
        Inputs are state and action batches as PyTorch tensors."""
    new_prob_batch = torch.zeros(action_batch.shape)
    row_index = 0
    for s,a in zip(state_batch, action_batch):
        new_prob_batch[row_index,:] = policy.calculate_log_probs_from_actions(s,a)
        row_index = row_index + 1
    # new_prob_list = [policy.calculate_log_probs_from_actions(s, a) for s, a in zip(state_list, action_list)]
    return new_prob_batch

def calculate_probability_ratio(old_prob_batch, new_prob_batch):
    """ Calculate the PPO probability ratio. The inputs old_prob_batch
    and new_prob_batch are expected to be N x 4 PyTorch tensors, with N
    being the number of samples in the batch."""
    assert(old_prob_batch.shape == new_prob_batch.shape)
    # Note: Need to collapse 4 probabilities (for 4 actions) into a scalar to 
    # multiply by the scalar rewards.  Done here by just summing the probabilities.
    # Note that they weren't really probabilities to begin with, but rather the
    # log of the normal distributions' PDF values.
    prob_ratio = torch.sum(torch.exp(new_prob_batch), axis=1) / torch.sum(torch.exp(old_prob_batch), axis=1)
    return prob_ratio

def clipped_surrogate(old_prob_batch, new_prob_batch, reward_batch,
                      discount=0.995,
                      epsilon=0.1,
                      beta=0.01):
    """ Calculate the PPO clipped surrogate function.  Inputs should be batches of
    training data, as PyTorch tensors. """
    prob_ratio = calculate_probability_ratio(old_prob_batch, new_prob_batch)
    clipped_prob_ratio = torch.clamp(prob_ratio, 1-epsilon, 1+epsilon)
    raw_loss = prob_ratio * reward_batch
    clipped_loss = clipped_prob_ratio * reward_batch
    ppo_loss = torch.min(raw_loss, clipped_loss)
    return ppo_loss

def calculate_entropy(old_prob_batch, new_prob_batch):
    old_prob_batch = torch.sum(torch.exp(old_prob_batch), axis=1)
    new_prob_batch = torch.sum(torch.exp(new_prob_batch), axis=1)
    entropy = -torch.exp(new_prob_batch) * (old_prob_batch + 1e-10) + \
              (1.0 - torch.exp(new_prob_batch)) * (1 - old_prob_batch + 1e-10)
    return entropy

def run_training_epoch(policy, optimizer, old_prob_list, state_list, action_list, reward_list,
                       discount=0.995,
                       epsilon=0.1,
                       beta=0.01,
                       batch_size=64):
    """ Run 1 training epoch.  Takes in the output lists from 1 run of collect_trajectories()
    for a single episode.  Breaks up the lists into batches and then runs the batches through 
    training. """
    num_samples = len(state_list)
    num_batches = int(np.ceil(num_samples/batch_size))
    sample_indices = np.arange(num_samples)

    old_prob_tensor = torch.tensor(np.concatenate(old_prob_list, axis=0))    # N x 4
    state_tensor = torch.tensor(np.concatenate(state_list, axis=0))          # N x 33
    action_tensor = torch.tensor(np.concatenate(action_list, axis=0))        # N x 4
    reward_tensor = torch.tensor(process_rewards(reward_list))               # N (1D)

    for batch_index in range(num_batches):
        sample_start_index = batch_index * batch_size
        sample_end_index = sample_start_index + batch_size
        batch_sample_indices = sample_indices[sample_start_index : sample_end_index]
        old_prob_batch = old_prob_tensor[batch_sample_indices,:]
        state_batch = state_tensor[batch_sample_indices,:]
        action_batch = action_tensor[batch_sample_indices,:]
        reward_batch = reward_tensor[batch_sample_indices]
        new_prob_batch = calculate_new_log_probs(policy, state_batch, action_batch)
    
        ppo_loss = clipped_surrogate(old_prob_batch, new_prob_batch, reward_batch,
                                     discount=discount, epsilon=epsilon, beta=beta)
        entropy = calculate_entropy(old_prob_batch, new_prob_batch)
        batch_loss = -torch.mean(ppo_loss + beta*entropy)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

