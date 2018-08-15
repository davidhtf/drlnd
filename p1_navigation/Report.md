[//]: # (Image References)

[image1]: https://github.com/davidhtf/drlnd/blob/master/p1_navigation/plot_of_rewards.png "Plot of Rewards"
[image2]: https://github.com/davidhtf/drlnd/blob/master/p1_navigation/plot_of_actions.png "Plot of Actions"

# Udacity DRLND - Project 1: Navigation

### Learning Algorithm

The learning algorithm for training an agent in this project is based on deep reinforcement learning - specifically it is built upon the Q-learning algorithm with a neural network of two fully connected hidden layers approximating the Q-value function. The neural network architecture is set out in `model.py` and the agent behavior is set out in `dqn_agent.py`.

The steps of the learning process can be summarize as follows: 
* For each episode, repeat the following:
  * For each step, repeat the following: 
    * After observing a state at time t, agent takes (epsilon-greedy) action per current policy 
    * After taking action at time t, agent observes state and reward at t+1
    * Agent then adds the "SARS" tuple to memory 
    * At every `UPDATE_EVERY` steps, agent then randomly select a batch of memory to update parameters of its Q-value function as approximated by the neural network as set out in `model.py` and as per the Q-learning algorithm
    * For the neural network optimization, we use mean-square-error as the loss function; gradients of each parameter of the neural network are backpropagated; and weight of each parameters are updated as per `optim.Adam`
    * At every `UPDATE_EVERY steps`, we also update the target Q-value function by performing `soft_update`, which generate an average of target Q-value function and local Q-value function with hyper-parameter `TAU` as the weight
    
The hyper-parameters of the learning algorithm include:
* Parameters concerning the epsilon greedy policy, which are:
  * Epsilon at start, epsilon decay rate and min epsilon
* Parameters concerning the agent object, which are:
  * Replay buffer size, minibatch size, discount factor, weight for soft update of target parameters, learning rate, and update steps for learning of the network
* Parameters concerning the neural network, which are:
  * Architecture of the neural network - in this case, the number of hidden layer and the number of node of each hidden layer

For tuning the agent to solving the environment (i.e. achieving an average score of 13+ over the last 100 episodes) with min training episodes needed. I have tried a permutation of the hyper-parameters set out above.

The best result I can obtain (solving the environemnt in 380 episodes) are achieve with the following setting:
* Epsilon at start = 1.0 , epsilon decay rate = 0.99, min epsilon = 0.05
* Replay buffer size = 1e5, minibatch size = 64, discount factor = 0.99, weight for soft update of target parameters = 1e-3, learning rate = 5e-4, update steps for learning of the network = 3
* Number of hidden layer = 2,  number of node of each hidden layer = {fc1_units: 64, fc2_units: 32}


### Plot Rewards

![Plot of Rewards][image1]

### Ideas for Future Work

![Plot of Actions][image2]
