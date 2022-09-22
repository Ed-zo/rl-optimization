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
        self.distances = torch.zeros((self.candidates, self.candidates), device = device)
        self.states = torch.zeros((self.count_of_envs, self.candidates), device = device)
        self.current_step = 0
        self.ordr = torch.arange(candidates, device = device).repeat((self.count_of_envs, 1))
        self.order = torch.arange(count_of_envs, device = device) * candidates

        arr = np.loadtxt(path + '/D.txt', delimiter=';')
        self.distances = torch.tensor(arr, device = device)

        self.max_actions = torch.tensor((count_of_envs, 1), device=device)

        f = open(path + '/C.txt', "r")
        lines = f.read().split('\n')
        f.close()

        for i in range(candidates):
            self.customers[0, i] = float(lines[i])

    def reset(self):
        self.states = torch.zeros((self.count_of_envs, self.candidates), device = self.device)
        self.prev_obj = torch.zeros((self.count_of_envs), device=self.device)
        self.max_actions = torch.zeros((self.count_of_envs, 1), device=self.device)
        self.current_step = 0
        return self.states.clone(), (1 - self.states)

    def compute_objective_function(self):
        if self.current_step == 0:
            return 0

        #True and False mask for whole columns
        built_mask = (self.states == 0.).repeat_interleave(self.candidates, 0)
        #Distance mask but repeated for every env
        dist = self.distances.repeat((self.count_of_envs, 1))
        #Set big number for non-built ones so Min doesnt pick them
        dist[built_mask] = 100000
        dist = dist.reshape((self.count_of_envs, self.candidates, self.candidates))
        #Find mins in every row
        mins = torch.min(dist, 2)
        #Objective function
        obj = mins.values * self.customers
        sum = obj.sum(1)

        return sum

    def step(self, actions):
        with record_function("Step"):
            self.states = self.states.view(-1)
            indices = self.order + actions.view(-1)
            self.states[indices] = 1
            self.states = self.states.view(-1, self.candidates)
            self.current_step += 1
            terminal = self.current_step == self.P
            
            # obj = self.compute_objective_function()

            # if(self.current_step == 1):
            #     rewards = torch.zeros((self.count_of_envs), device=self.device)
            # # elif(terminal):
            # #     rewards = obj / -10000
            # else:
            #     rew = (self.prev_obj - obj)
            #     rewards = rew / 10000

            # self.prev_obj = obj

            if (terminal):
                m = self.states.sum(1) == self.P
                rewards = (self.compute_objective_function() / -100000) * m
                rewards = rewards - torch.ones((self.count_of_envs), device=self.device) * ~m
            else:
                rewards = torch.zeros((self.count_of_envs), device=self.device)

            self.max_actions = torch.max(self.max_actions, actions)

            mask = self.ordr >= self.max_actions

            return self.states.clone(), mask, rewards, terminal
