import torch

class Env:
    def __init__(self, p, candidates, count_of_envs, path, device):
        self.P = p
        self.candidates = candidates
        self.count_of_envs = count_of_envs
        self.device = device
        self.customers = torch.zeros((1, self.candidates), device = device)
        self.distances = torch.zeros((1, self.candidates, self.candidates), device = device)
        
        #two matrices - first is distance matrix and second is if "sklad" is built
        self.states = torch.zeros((self.count_of_envs, 2, self.candidates, self.candidates), device = device)
        self.built = torch.zeros((self.count_of_envs, self.candidates), device = device)
        self.current_step = 0
        self.order = torch.arange(count_of_envs, device = device) * candidates
        self.prev_obj = torch.zeros((self.count_of_envs))

        f = open(path + '/D.txt', "r")
        lines = f.read().split('\n')
        f.close()

        for i in range(candidates):
            values = lines[i].split(';')
            for j in range(candidates):
                self.distances[0, i, j] = float(values[j])
                self.states[:, 0, i, j] = float(values[j])

        f = open(path + '/C.txt', "r")
        lines = f.read().split('\n')
        f.close()

        for i in range(candidates):
            self.customers[0, i] = float(lines[i])

        #self.customers = self.customers.repeat(self.count_of_envs, 1)
        #self.distances = self.distances.repeat(self.count_of_envs, 1, 1)

    def reset(self):
        self.built = torch.zeros((self.count_of_envs, self.candidates))
        self.states[:, 1, :, :] = 0
        self.prev_obj = torch.zeros((self.count_of_envs), device=self.device)
        self.current_step = 0
        
        return self.states.clone(), (1 - self.built.clone()).abs()

    def compute_objective_function(self):
        if self.current_step == 0:
            return 0

        objective_values = torch.zeros((self.count_of_envs))
        built_cpu = self.built.cpu()
        for i in range(self.count_of_envs):
            placements = torch.nonzero(built_cpu[i]).squeeze(1)
            mins = torch.min(self.distances.index_select(1, placements), 1)
            sum = (mins.values * self.customers).sum()
            objective_values[i] = sum.item()

        return objective_values.to(self.device)

    def step(self, actions):
        self.built = self.built.view(-1)
        indices = self.order + actions.view(-1)
        self.built[indices] = 1
        self.built = self.built.view(-1, self.candidates)

        for i in range(len(actions)):
            self.states[i, 1, :, actions[i]] = 1
            self.states[i, 1, actions[i], :] = 1
            
        self.current_step += 1
        terminal = self.current_step == self.P
        
        if(terminal):
            rewards = (self.compute_objective_function() / -1000) * terminal
        else:
            rewards = torch.zeros((self.count_of_envs), device=self.device)

        return self.states.clone(), (1 - self.built.clone()).abs(), rewards, terminal
