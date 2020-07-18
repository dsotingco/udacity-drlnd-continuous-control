# udacity-drlnd-continuous-control

# Introduction
This is my submission for the Reacher (Continuous Control) project of the Udacity Deep Reinforcement Learning Nanodegree.  The purpose of the project is to train a group of 20 agents to each maintain a two-link robotic arm at its target position.  I chose to use Proximal Policy Optimization (PPO) to solve this project.

# Project Details
The environment for this project is a Unity ML-Agents environment provided by Udacity.  In this environment. each of 20 identical agents controls a double-jointed arm.  Shown in the environment are blue balls indicating the target location for each.  The blue balls turn green when their corresponding arms are in the target location for a particular timestep.  Each agent is provided a reward of 0.1 for each timestep that its end-effector is in the target location.  Therefore, each agent's goal is to maintain its position at the target location for as long as possible.

The state space of each agent's observations is a vector of 33 numbers corresponding to position, rotation, velocity, and angular velocities of the arm.  Each agent's action space consists of 4 numbers between -1 and 1, corresponding to torque applicable to the arm's two joints.

The environment is considered solved when the agents get an average score of +30 over 100 consecutive episodes, and over all agents.

# Getting Started
The dependencies for this submission are the same as for the [Udacity Deep Reinforcement Learning nanodegree](https://github.com/udacity/deep-reinforcement-learning#dependencies):
* Python 3.6
* pytorch
* unityagents
* [Udacity 20-agent Reacher environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

This project was completed and tested on a local Windows environment (64-bit) running Python 3.6.10 in a conda environment.

# Instructions
To train the agent, run **train_reacher_agent.py**.  This will save the model parameters in **reacher_weights.pth** once the agent has fulfilled the criteria for considering the environment solved.

To run the trained agent, run **run_reacher_agent.py**.  This will load the saved model parameters from reacher_weights.pth, and run the trained model in the Reacher environment.  The *n_episodes* parameter is the number of episodes that will be run.  By default, this parameter is set to 100 to facilitate validation of the trained agent.

