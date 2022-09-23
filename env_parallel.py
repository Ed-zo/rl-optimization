import torch

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
        self.order = torch.arange(count_of_envs, device = device) * candidates

        f = open(path + '/D.txt', "r")
        lines = f.read().split('\n')
        f.close()

        for i in range(candidates):
            values = lines[i].split(';')
            for j in range(candidates):
                self.distances[i, j] = float(values[j])

        f = open(path + '/C.txt', "r")
        lines = f.read().split('\n')
        f.close()

        for i in range(candidates):
            self.customers[0, i] = float(lines[i])

    def reset(self):
        self.states = torch.zeros((self.count_of_envs, self.candidates), device = self.device)
        self.prev_obj = torch.zeros((self.count_of_envs), device=self.device)
        self.current_step = 0
        return self.states.clone()

    def compute_objective_function(self):
        if self.current_step == 0:
            return 0

        #True and False mask for whole columns
        built_mask = (self.states == 0.).repeat_interleave(self.candidates, 0)
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
        self.states = self.states.view(-1)
        indices = self.order + actions.view(-1)
        self.states[indices] = 1
        self.states = self.states.view(-1, self.candidates)
        self.current_step += 1
        terminal = self.current_step == self.P
        
        # obj = self.compute_objective_function()

        # if(self.current_step == 1):
        #     rewards = torch.zeros((self.count_of_envs), device=self.device)
        # elif(terminal):
        #     rewards = obj / -1000
        # else:
        #     rew = (self.prev_obj - obj)
        #     rewards = rew / 10000

        # self.prev_obj = obj

        if(terminal):
            rewards = (self.compute_objective_function() / -100000) * terminal
        else:
            rewards = torch.zeros((self.count_of_envs), device=self.device)

        return self.states.clone(), rewards, terminal
