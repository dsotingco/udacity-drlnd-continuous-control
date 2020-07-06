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
        (actions, probs) = policy.forward(states)
        env_info = env.step(actions.detach().numpy())[brain_name]
        next_states = env_info.vector_observations.astype(np.float32)
        rewards = np.array(env_info.rewards)
        dones = env_info.local_done
        scores += env_info.rewards
        # Append results to output lists.
        assert isinstance(probs, torch.Tensor)
        assert isinstance(states, np.ndarray)
        assert isinstance(actions, torch.Tensor)
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
    num_timesteps rows, and num_agents columns."""
    # calculate discounted rewards
    discount_array = discount ** np.arange(len(reward_list))
    discounted_rewards = np.asarray(reward_list) * discount_array[:,np.newaxis]

    # calculate future discounted rewards
    future_rewards = discounted_rewards[::-1].cumsum(axis=0)[::-1]

    # normalize the future discounted rewards
    mean = np.mean(future_rewards, axis=1)
    std = np.std(future_rewards, axis=1) + 1.0e-10
    normalized_rewards = (future_rewards - mean[:,np.newaxis]) / std[:,np.newaxis]
    return normalized_rewards

def calculate_new_log_probs(policy, state_batch, action_batch):
    """ Calculate new log probabilities of the actions, 
        given the states.  To be used during training as the
        policy is changed by the optimizer. 
        Inputs are state and action batches as PyTorch tensors."""
    new_prob_batch = policy.calculate_log_probs_from_actions(state_batch, action_batch)
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
    old_log_probs_summed = torch.sum(old_prob_batch, dim=2)
    new_log_probs_summed = torch.sum(new_prob_batch, dim=2)
    prob_ratio = torch.exp(new_log_probs_summed - old_log_probs_summed)
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
    old_prob_batch = torch.sum(torch.exp(old_prob_batch))
    new_prob_batch = torch.sum(torch.exp(new_prob_batch))
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
    np.random.shuffle(sample_indices)

    old_prob_tensor = torch.stack(old_prob_list).detach()                           # 1001 x 20 x  4 (T x num_agents x action_size)
    state_tensor = torch.tensor(state_list, dtype=torch.float).detach()             # 1001 x 20 x 33 (T x num_agents x state_size)
    action_tensor = torch.stack(action_list).detach()                               # 1001 x 20 x  4 (T x num_agents x action_size)
    reward_tensor = torch.tensor(process_rewards(reward_list), dtype=torch.float)   # 1001 x 20      (T x num_agents)

    for batch_index in range(num_batches):
        sample_start_index = batch_index * batch_size
        sample_end_index = sample_start_index + batch_size
        batch_sample_indices = sample_indices[sample_start_index : sample_end_index]

        old_prob_batch = old_prob_tensor[batch_sample_indices]
        state_batch = state_tensor[batch_sample_indices]
        action_batch = action_tensor[batch_sample_indices]
        reward_batch = reward_tensor[batch_sample_indices]
        new_prob_batch = calculate_new_log_probs(policy, state_batch, action_batch)

        print("old_prob_batch.shape: ", old_prob_batch.shape)
        print("state_batch.shape: ", state_batch.shape)
        print("action_batch.shape: ", action_batch.shape)
        print("reward_batch.shape: ", reward_batch.shape)
        print("new_prob_batch.shape: ", new_prob_batch.shape)
    
        ppo_loss = clipped_surrogate(old_prob_batch, new_prob_batch, reward_batch,
                                     discount=discount, epsilon=epsilon, beta=beta)
        entropy = calculate_entropy(old_prob_batch, new_prob_batch)
        batch_loss = -torch.mean(ppo_loss + beta*entropy)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

