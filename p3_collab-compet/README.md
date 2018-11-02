[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

The following is built upon the original README from the [Udacity DRLND](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)

### Project Details

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

My model was trained locally on a Dell XPS 15 9560 with GPU enabled under a Window (64-bit) operating system. Thus, the following steps and instrcuctions are specifically for Window (64-bit).

1. To set up the required Python environment, follow the steps set out under **Dependencies** from [this link](https://github.com/davidhtf/drlnd)   

2. The Unity environment has been uploaded to this repository so there is no need to download it if you have cloned this repository

### Instructions

- To train an agent, follow the instructions set out in `train.ipynb`