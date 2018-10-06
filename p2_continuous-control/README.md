[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Udacity DRLND - Project 2: Continuous Control

### Introduction

The following is built upon the original README from the [Udacity DRLND](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control)

### Project Details

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

Here we aim to solve the first option (single agent) of the problem: "The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes."

### Getting Started

My model has been trained locally on a Dell XPS 15 9560 with GPU enabled under a Window (64-bit) operating system. Thus, the following steps and instrcuctions are specifically for Window (64-bit).

1. To set up the required Python environment, follow the steps set out under **Dependencies** from [this link](https://github.com/davidhtf/drlnd)   

2. The Unity environment has been uploaded to this repository so there is no need to download it if you have cloned this repository

### Instructions

- To train an agent, follow the instructions set out in `train.ipynb`
