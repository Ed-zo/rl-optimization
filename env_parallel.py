import torch

class Env:
    def __init__(self, p, candidates, count_of_envs, path, device):
        self.P = p
        self.candidates = candidates
        self.count_of_envs = count_of_envs
        self.device = device
        self.customers = torch.zeros((1, self.candidates), device=device)
        self.distances = torch.zeros((self.candidates, self.candidates), device=device)
        
        #distance matrix, customer needs, nearest built sklad, matrix if "sklad" is built
        self.states = torch.zeros((self.count_of_envs, 4, self.candidates, self.candidates), device = device)
        self.built = torch.zeros((self.count_of_envs, self.candidates), device = device)
        self.current_step = 0
        self.order = torch.arange(count_of_envs, device = device) * candidates

        f = open(path + '/D.txt', "r")
        lines = f.read().split('\n')
        f.close()

        for i in range(candidates):
            values = lines[i].split(';')
            for j in range(candidates):
                self.distances[i, j] = float(values[j])
                self.states[:, 0, i, j] = float(values[j])

        max_D = self.states[0, 0, :, :].max()
        self.states[:, 0, :, :] = self.states[:, 0, :, :] / max_D

        f = open(path + '/C.txt', "r")
        lines = f.read().split('\n')
        f.close()
        
        for i in range(candidates):
            self.customers[0, i] = float(lines[i])

        max_C = self.customers.max(dim=-1).values
        self.states[:, 1, :, :] = self.customers[0] / max_C

        self.states[:, 2] = -1

    def reset(self):
        self.built = torch.zeros((self.count_of_envs, self.candidates), device=self.device)
        self.states[:, 3, :, :] = 0
        self.states[:, 2] = -1
        self.current_step = 0
        
        return self.states.clone(), (1 - self.built.clone())

    def compute_mins(self):
        #True and False mask for whole columns
        built_mask = (self.built == 0.).repeat_interleave(self.candidates, 0)
        #Distance mask but repeated for every env
        dist = self.distances.repeat((self.count_of_envs, 1))
        #Set big number for non-built ones so Min doesnt pick them
        dist[built_mask] = 10000
        dist = dist.reshape((self.count_of_envs, self.candidates, self.candidates))
        #Find mins in every row
        mins = torch.min(dist, 2)

        return mins

    def compute_objective_function(self, mins = None):
        if self.current_step == 0:
            return 0

        if mins is None:
            mins = self.compute_mins()
        
        #Objective function
        obj = mins.values * self.customers
        sum = obj.sum(1)

        return sum

    def step(self, actions):
        self.built = self.built.view(-1)
        indices = self.order + actions.view(-1)
        self.built[indices] = 1
        self.built = self.built.view(-1, self.candidates)

        s_actions = actions.squeeze()
        ar = torch.arange(self.count_of_envs)
        self.states[ar, 3, :, s_actions] = 1
        self.states[ar, 3, s_actions, :] = 1
            
        self.current_step += 1
        terminal = self.current_step == self.P

        mins = self.compute_mins()
        x = (mins.indices / self.candidates).repeat_interleave(self.candidates, 0)
        self.states[:, 2] = x.reshape((self.count_of_envs, self.candidates, self.candidates))
        
        if(terminal):
            rewards = (self.compute_objective_function(mins) / -100000) * terminal
        else:
            rewards = torch.zeros((self.count_of_envs), device=self.device)

        return self.states.clone(), (1 - self.built.clone()), rewards, terminal
