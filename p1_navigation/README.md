[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Udacity DRLND - Project 1: Navigation

The following is modified from the original README.md form the Udacity DRLMD: (https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)

### Project Details

In this project, we are asked to train an agent to play BananaCollectors (which is derived from the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)) 

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes. Each episode will end when 300 steps are taken.

### Getting Started

The model whas been trained locally on a Dell XPS 15 9560 with GPU enabled under a Window (64-bit) operating system. Thus, the following steps and instrcuctions are specifically for Window (64-bit).

1. To set up the Python environment needed to for running the codes, follow the steps set out under **Dependencies** from this link: (https://github.com/davidhtf/drlnd)   

2. The Unity environment has been uploaded to this repository so there is no need to download it if you have cloned this repository

### Instructions

- To train an agent, follow the instructions set out in `train_dqn.ipynb`
- To test an agent (i.e. see how a trained agent plays), follow the instructions set out in `test_dqn.ipynb`
