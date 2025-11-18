"""
DQN Agent and Replay Buffer Implementation
"""

import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Experience Replay Buffer for DQN"""
    
    def __init__(self, capacity=100000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent
    
    Implements epsilon-greedy policy for action selection
    """
    
    def __init__(self, state_dim, action_dim, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995):
        """
        Initialize DQN agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def select_action(self, state, policy_net, device, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state observation
            policy_net: Policy network
            device: Torch device (cpu/cuda)
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        import torch
        
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_dim)
        else:
            # Exploit: greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                return q_values.argmax().item()
    
    def update_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_epsilon(self):
        """Get current epsilon value"""
        return self.epsilon
