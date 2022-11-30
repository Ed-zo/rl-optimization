import torch
import numpy as np
from torch.profiler import record_function

class Env:
    def __init__(self, p, candidates, count_of_envs, path, device):
        self.P = p
        self.candidates = candidates
        self.count_of_envs = count_of_envs
        self.device = device
        self.customers = torch.zeros((1, self.candidates), device = device)
        self.ordr = torch.arange(candidates, device = device).repeat((self.count_of_envs, 1))
        self.order = torch.arange(count_of_envs, device = device) * candidates

        arr = np.loadtxt(path + '/D.txt', delimiter=';')
        self.distances = torch.tensor(arr, device = device, dtype=torch.int16)

        f = open(path + '/C.txt', "r")
        lines = f.read().split('\n')
        f.close()

        for i in range(candidates):
            self.customers[0, i] = float(lines[i])

    def reset(self):
        self.states = torch.zeros((self.count_of_envs, self.candidates * 2 + 1), device = self.device)
        self.build_count = torch.zeros((self.count_of_envs), device=self.device)
        self.current_step = 0

        self.build_order = torch.zeros((self.count_of_envs, self.candidates), dtype=torch.long, device=self.device)
        for i in range(self.count_of_envs):
            self.build_order[i] = torch.randperm(self.candidates, device=self.device)
            # self.build_order[i] = torch.arange(self.candidates, device=self.device)

        indices = self.build_order[:, 0] + self.candidates
        self.states[torch.arange(0, self.count_of_envs), indices] = 1

        return self.states.clone()

    def compute_objective_function(self):
        #True and False mask for whole columns
        built_mask = (self.states[:, :self.candidates] == 0.).repeat_interleave(self.candidates, 0)
        #Distance mask but repeated for every env
        dist = self.distances.repeat((self.count_of_envs, 1))
        #Set big number for non-built ones so Min doesnt pick them
        dist[built_mask] = 10000
        dist = dist.reshape((self.count_of_envs, self.candidates, self.candidates))
        #Find mins in every row
        mins = torch.min(dist, 2)
        #Objective function
        obj = mins.values * self.customers
        sum = obj.sum(1)

        return sum

    def step(self, actions):
        # self.states = self.states.view(-1)
        # indices = self.build_order[:, self.current_step]
        # self.states[indices] = 1
        # self.states = self.states.view(-1, self.candidates)

        env_order = torch.arange(0, self.count_of_envs)
        indices = self.build_order[:, self.current_step]
        self.states[env_order, indices] = (1 - self.states[env_order, -1]) * actions.view(-1).type(torch.float)
        self.states[env_order, indices + self.candidates] = 0
        self.current_step += 1

        self.build_count += actions.view(-1)
        
        build_terminal = self.build_count >= self.P
        self.states[build_terminal, -1] = 1

        terminal = self.current_step == self.candidates

        if terminal:
            rewards = (self.compute_objective_function() / -100000) - (~build_terminal * 1)
        else:
            rewards = torch.zeros((self.count_of_envs), device=self.device)
            indices = self.build_order[:, self.current_step]
            self.states[env_order, indices + self.candidates] = 1

        return self.states.clone(), rewards, terminal
