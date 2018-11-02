from ddpg_agent import Agent

import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque

BUFFER_SIZE = int(1e5)  # replay buffer size,                       benchmark: int(1e5)
BATCH_SIZE = 256        # minibatch size,                           benchmark: 256
GAMMA = 1.0             # discount factor,                          benchmark: 1.0
TAU = 5e-3              # for soft update of target parameters,     benchmark: 5e-3
UPDATE_EVERY = 2        # steps for each update of parameters,      benchmark: 2
NOISE_REDUCTION = 1     # for reduction of noise applied to action, benchmark: 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MMDDPG():
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of DDPG agents in total
            random_seed (int): random seed
        """

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

        # Initialize multiple DDPG agents 
        self.agents = [Agent(state_size=state_size, action_size=action_size, 
            num_agents=num_agents, random_seed=random_seed) for i in range(num_agents)]
         
        # Initialize replay memory (to be shared by all agents)
        self.memory = ReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=random_seed)
        
        # for scaling down noise applied to action as episode increases
        self.noise_scale = 1.0
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, states):
        """Returns actions of each agents for given states as per current policy"""

        self.noise_scale *= NOISE_REDUCTION

        actions = [agent.act(state, self.noise_scale) for agent, state in zip(self.agents, states)]
        
        return np.array(actions)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        #reshape states and next states observed by all agents
        full_states = states.reshape(-1)
        full_next_states = next_states.reshape(-1)

        #save experiences
        self.memory.add(state=states, full_state=full_states, action=actions, reward=rewards,
            next_state=next_states, full_next_state=full_next_states, done=dones)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                # each agent learn, if enough samples are available in memory
                for agent_id in range(self.num_agents):
                    experiences = self.memory.sample()
                    self.learn(agent_id, experiences, GAMMA)

    def learn(self, agent_id, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            agent_id (int): Id of the DDPG agent to learn
            experiences (Tuple[torch.Tensor]): tuple of (s, s_all a, r, s', s'_all, done) tuples 
            gamma (float): discount factor
        """
        
        agent = self.agents[agent_id]
        
        # Dim of each element in a experience tuples
        # dim(s_states) = BATCH_SIZE x num_agents x state_size
        # dim(s_full_states) = BATCH_SIZE x 1 x (state_size * num_agents)
        # dim(s_actions) = BATCH_SIZE x num_agents x action_size
        # dim(s_rewards) = BATCH_SIZE x num_agents
        # dim(s_next_states) = BATCH_SIZE x num_agents x state_size
        # dim(s_full_next_states) = BATCH_SIZE x 1 x (state_size * num_agents)
        # dim(s_dones) = BATCH_SIZE x num_agents

        s_states, s_full_states, s_actions, s_rewards, s_next_states, s_full_next_states, s_dones = experiences
        
        rewards = s_rewards[:, agent_id].view(-1, 1)
        dones = s_dones[:, agent_id].view(-1, 1)

        # ---------------------------- update critic ---------------------------- #

        # Get predicted next-state actions for all agents
        actions_next = torch.zeros(s_next_states.shape[:2] + (self.action_size,), dtype=torch.float, device=device)        
        
        for i in range(self.num_agents):
            actions_next[:, i, :] = self.agents[i].actor_target(s_next_states[:, i])

        actions_next = actions_next.view(-1, self.action_size * self.num_agents)

        # Compute Q targets with next-state observaion and next-state actions for all agents
        Q_targets_next = agent.critic_target(s_full_next_states, actions_next)
        
        # Compute Q targets for current-state
        Q_targets = rewards + (gamma * Q_targets_next * (1.0 - dones))
        
        # Compute Q local for current-state
        actions = s_actions.view(-1, self.action_size * self.num_agents)
        Q_expected = agent.critic_local(s_full_states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        
        # states observed by the agent being trained only
        states = s_states[:, agent_id]
        
        # update the agent being trained only
        agent_actions = agent.actor_local(states)
        actions_pred = s_actions.clone()
        actions_pred[:, agent_id] = agent_actions

        actions_pred = actions_pred.view(-1, self.action_size * self.num_agents)
        
        # Compute actor loss
        actor_loss = -agent.critic_local(s_full_states, actions_pred).mean()
        
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target, TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, TAU)   

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "full_state", "action", "reward", "next_state", "full_next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, full_state, action, reward, next_state, full_next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, full_state, action, reward, next_state, full_next_state, done)
        
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.array([e.state for e in experiences if e is not None])
        states = torch.from_numpy(states).float().to(device)

        full_states = np.array([e.full_state for e in experiences if e is not None])
        full_states = torch.from_numpy(full_states).float().to(device)

        actions = np.array([e.action for e in experiences if e is not None])
        actions = torch.from_numpy(actions).float().to(device)

        rewards = np.array([e.reward for e in experiences if e is not None])
        rewards = torch.from_numpy(rewards).float().to(device)

        next_states = np.array([e.next_state for e in experiences if e is not None])
        next_states = torch.from_numpy(next_states).float().to(device)

        full_next_states = np.array([e.full_next_state for e in experiences if e is not None])
        full_next_states = torch.from_numpy(full_next_states).float().to(device)

        dones = np.array([e.done for e in experiences if e is not None])
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(device)

        return (states, full_states, actions, rewards, next_states, full_next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)