[//]: # (Image References)

[image1]: https://github.com/davidhtf/drlnd/blob/master/p2_continuous-control/plot_of_rewards.png "Plot of Rewards"

# Udacity DRLND - Project 2: Continuous Control

### Learning Algorithm

The learning algorithm for training an agent in this project is based on deep reinforcement learning - specifically it is built upon the DDPG algorithm with one neural network approximating the value function (critic) and another one approximating the policy function (actor). The neural network architectures are set out in `model.py` and the agents' behavior are set out in `ddpg_agent.py`.

The steps of the learning process can be summarize as follows: 
* For each episode, repeat the following:
  * For each step, repeat the following: 
    * After observing a state at time t, agent takes action per current policy (we also clip the action such that abs(action) <= 1, and allow for random noise per Ornstein-Uhlenbeck process)
    * After taking action at time t, agent observes state and reward at t+1
    * Agent then adds the "SARS" tuple to memory 
    * At every `UPDATE_EVERY` steps, agent then randomly select a batch of memory to update parameters of its value function and policy function as approximated by the neural networks as set out in `model.py` and as per the DDPG algorithm
    * For the neural networks' optimization, we use mean-square-error as the loss function; gradients of each parameter of the neural network are backpropagated; and weight of each parameters are updated as per `optim.Adam`
    * At every `UPDATE_EVERY steps`, we also update the targets of the two functions by performing `soft_update`, which generate an average of target functions and local functions with hyper-parameter `TAU` as the weight
    
The hyper-parameters of the learning algorithm include:
* Parameters concerning the agent object, which are:
  * Replay buffer size, minibatch size, discount factor, weight for soft update of target parameters, learning rates, L2 weight decay, and update steps for learning of the network
* Parameters concerning the neural networks, which are:
  * Architecture of the neural network - in this case, for each function, the number of hidden layer and the number of node of each hidden layer
* Parameters concerning the Ornstein-Uhlenbeck process: mu, theta and sigma
* Maximum number of steps allows for each episode

For tuning the agent to solving the environment (i.e. achieving an average reward of 30+ over the last 100 episodes) with min training episodes needed. I have tried a permutation of the hyper-parameters set out above.

The best (also considering stability of convergence) result I can obtain (solving the environemnt in 1421 episodes) are achieved with the following setting:
* Replay buffer size = 1e6, minibatch size = 128, discount factor = 0.99, weight for soft update of target parameters = 1e-3, learning rate of actor = 1e-4, learning rate of critic = 1e-4 , L2 weight decay = 1e-4, update steps for learning of the network = 10
* Number of hidden layer (actor) = 2,  number of node of each hidden layer (actor) = {fc1_units: 128, fc2_units: 64}; Number of hidden layer (critic) = 2,  number of node of each hidden layer (critic) = {fc1_units: 256, fc2_units: 128}
* Parameters concerning the Ornstein-Uhlenbeck process: mu = 0, theta = 0.15 and sigma = 0.1
* Maximum number of steps allows for each episode = 1000


### Plot Rewards

Set forth below is a plot of rewards per epsiode showing that trained agaent is able to receive an average reward (over 100 episodes) of at least 30.

![Plot of Rewards][image1]


### Ideas for Future Work

The current set up (i.e. DDPG with a single agent) is able to solve the environment with a relatively stable convergence. However the number of episodes required seems large. I note that other fellow classmates of the DRLDN program are able to 1) solve the environment faster with training multiple agents at the same time and 2) obtain more stable convergence by using PPO, and thus I will also explore these alternatives as for next steps.  
