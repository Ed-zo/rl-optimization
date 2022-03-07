import numpy as np
import torch

class ExperienceReplay():
    def __init__(self, size, state_dim):
        self.index = 0
        self.size = size
        self.state_dim = state_dim
        dim = (size, ) + state_dim

        self.states = torch.zeros(dim)
        self.actions = torch.zeros((self.size), dtype=torch.int8)
        self.rewards = torch.zeros(self.size)
        self.states_ = torch.zeros(dim)
        self.terminals = torch.zeros(self.size)

    def store(self, state, action, reward, state_, terminal):
        index = self.index % self.size

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.states_[index] = state_
        self.terminals[index] = int(terminal)

        self.index += 1

    def sample(self, batch_size):
        length = min(self.size, self.index)

        batch = np.random.choice(length, batch_size)
        states =  self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        states_ = self.states_[batch]
        terminal = self.terminals[batch]

        return states, actions, rewards, states_, terminal