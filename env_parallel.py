import torch
import numpy as np

class Env:
    def __init__(self, p, candidates, count_of_envs, path, device):
        self.P = p
        self.candidates = candidates
        self.count_of_envs = count_of_envs
        self.device = device
        self.customers = torch.zeros((1, self.candidates), device = device)
        self.distances = torch.zeros((self.candidates, self.candidates), device = device)
        self.states = torch.zeros((self.count_of_envs, self.candidates * 2), device = device)
        self.current_step = 0
        self.buildOrder = torch.arange(count_of_envs, device = device) * candidates
        self.actionOrder = torch.arange(count_of_envs, device = device) * candidates * 2

        arr = np.loadtxt(path + '/D.txt', delimiter=';')
        self.distances = torch.tensor(arr, device = device)

        f = open(path + '/C.txt', "r")
        lines = f.read().split('\n')
        f.close()

        for i in range(candidates):
            self.customers[0, i] = float(lines[i])

        self.states[:, self.candidates:] = -1

    def reset(self):
        self.states = torch.zeros((self.count_of_envs, self.candidates * 2), device = self.device)
        self.states[:, self.candidates:] = -1
        self.current_step = 0

        self.dist = self.distances.repeat((self.count_of_envs, 1)) + 100

        return self.states.clone(), (1 - self.states[:, :self.candidates])

    def compute_mins(self):
        #True and False mask for whole columns
        built_mask = (self.states[:, :self.candidates] == 0.).repeat_interleave(self.candidates, 0)
        #Distance mask but repeated for every env
        dist = self.distances.repeat((self.count_of_envs, 1))
        #Set big number for non-built ones so Min doesnt pick them
        dist[built_mask] = 100000
        dist = dist.reshape((self.count_of_envs, self.candidates, self.candidates))
        #Find mins in every row
        mins = torch.min(dist, 2)

        return mins

    def compute_objective_function(self, mins):
        if self.current_step == 0:
            return 0

        #Objective function
        obj = mins.values * self.customers
        sum = obj.sum(1)

        return sum

    def step(self, actions):
        self.states = self.states.view(-1)
        indices = self.actionOrder + actions.view(-1)
        self.states[indices] = 1
        self.states = self.states.view(-1, self.candidates * 2)
        self.current_step += 1
        terminal = self.current_step == self.P
        
        indices = self.buildOrder + actions.view(-1)
        self.dist[indices] -= 100
        s = self.dist.reshape((self.count_of_envs, self.candidates, self.candidates))
        mins = torch.min(s, 1)

        # mins = self.compute_mins()
        self.states[:, self.candidates:] = mins.indices / self.candidates

        if(terminal):
            obj = self.compute_objective_function(mins)
            rewards = obj / -100000
        else:
            obj = None
            rewards = torch.zeros((self.count_of_envs), device=self.device)

        return self.states.clone(), (1 - self.states[:, :self.candidates]), rewards, terminal, {'obj': obj}
