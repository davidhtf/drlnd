[//]: # (Image References)

[image1]: https://github.com/davidhtf/drlnd/blob/master/p3_collab-compet/plot_of_max_scores.png "Plot of Rewards"

# Udacity DRLND - Project 3: Collaboration and Competition

### Learning Algorithm

The learning algorithm for training an agent in this project is based on deep reinforcement learning - specifically it is built upon the MADDPG algorithm with two each DDPG agents sharing the same replay experience. The neural network architectures are set out in `model.py` and the agents' behavior are set out in `maddpg.py` and `ddpg_agent.py` (these are extensions from what have been developed for [Projec Two](https://github.com/davidhtf/drlnd/tree/master/p2_continuous-control)

The steps of the learning process can be summarize as follows: 
* For each episode, repeat the following:
  * For each step, repeat the following: 
    * After observing a state at time t, each agent takes action per current policy (we also clip the action such that abs(action) <= 1, but unlike in Project 2, with noise disabled)
    * After taking action at time t, each agent observes its state and reward at t+1
    * Agent then adds the "S, S_all, A, R, S', S'_all" tuple to memory where is to be shared by each agent; the dimension of each element of the tuple are as such:
        * dim(s_states) = BATCH_SIZE x num_agents x state_size
        * dim(s_full_states) = BATCH_SIZE x 1 x (state_size * num_agents)
        * dim(s_actions) = BATCH_SIZE x num_agents x action_size
        * dim(s_rewards) = BATCH_SIZE x num_agents
        * dim(s_next_states) = BATCH_SIZE x num_agents x state_size
        * dim(s_full_next_states) = BATCH_SIZE x 1 x (state_size * num_agents)
        * dim(s_dones) = BATCH_SIZE x num_agents              
    * At every `UPDATE_EVERY` steps, each agent then randomly select a batch of memory to update parameters of its value function and policy function as approximated by the neural networks as set out in `model.py` and as per the MADDPG algorithm (that is, the Critic is trained with obervations and actions of ALL agents)
    * For the neural networks' optimization, we use mean-square-error as the loss function; gradients of each parameter of the neural network are backpropagated; and weight of each parameters are updated as per `optim.Adam`
    * At every `UPDATE_EVERY steps`, for each agent, we also update the targets of the two functions by performing `soft_update`, which generate an average of target functions and local functions with hyper-parameter `TAU` as the weight
    
The hyper-parameters of the learning algorithm include:
* Parameters concerning the agent object, which are:
  * Replay buffer size, minibatch size, discount factor, weight for soft update of target parameters, learning rates, L2 weight decay, and update steps for learning of the network
* Parameters concerning the neural networks, which are:
  * Architecture of the neural network - in this case, for each function, the number of hidden layer and the number of node of each hidden layer
* Parameters concerning the Ornstein-Uhlenbeck process: mu, theta and sigma
* Maximum number of steps allows for each episode

For tuning the agent to solving the environment (i.e. achieving an average reward of 0.5+ over the last 100 episodes) with min training episodes needed. I have tried a permutation of the hyper-parameters set out above.

The best (also considering stability of convergence) result I can obtain (solving the environemnt in 2340 episodes) are achieved with the following setting:
* Replay buffer size = 1e5, minibatch size = 256, discount factor = 1.0, weight for soft update of target parameters = 5e-3, learning rate of actor = 1e-4, learning rate of critic = 1e-4 , L2 weight decay = 0.0, update steps for learning of the network = 2
* Number of hidden layer (actor) = 2,  number of node of each hidden layer (actor) = {fc1_units: 256, fc2_units: 128}; Number of hidden layer (critic) = 2,  number of node of each hidden layer (critic) = {fc1_units: 512, fc2_units: 256}
* Parameters concerning the Ornstein-Uhlenbeck process: N/M as we have disabled noise
* Maximum number of steps allows for each episode = 1000


### Plot Rewards

Set forth below is a plot of rewards per epsiode showing that trained agaent is able to receive an average reward (over 100 episodes) of at least +0.5.

![Plot of Rewards][image1]


### Ideas for Future Work

The result of applying MADDPG in this case is highly sensitive to the changes of hyperparameters, and I have had difficulty to get the model start learning at the very beginning. Given this is a symmetric zero-sum game, I will try to apply the AlphaZero algorithm set out in the last part of the Nanodegree to see if a better (i.e. faster and/or more stable training) result can be obtained.    
