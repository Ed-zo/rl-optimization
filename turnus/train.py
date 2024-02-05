import inspect
import torch
import torch.nn as nn
import datetime
import torch.nn.functional as F
from ppo_parallel import Agent
import torch.nn.init as init
import numpy as np
from env import Env
import utils.graph_utils as graph_utils
from model import GCNPolicy, RNDModel

# print(inspect.getfile(GCN))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    start_date = datetime.datetime.now()
    print('Start time:', start_date)

    problem = 2
    problem_path = f'data/{problem}'

    graph, optimal_vehicles = graph_utils.load_problem(problem_path)

    env = Env(graph, device, optimal_vehicles)

    net = GCNPolicy(env.state_space(), env.action_space()).to(device)
    rnd_net = RNDModel(env.state_space(), env.action_space()).to(device)
    net.train()
    rnd_net.train()

    agent = Agent(net, rnd_net, device=device, name='ppo_problem_2', path='results/', ext_gamma=1, epsilon=0.2, lr=0.001)

    agent.train([graph], Env, graph.num_nodes, count_of_iterations=10000, count_of_processes=2, count_of_envs=16, 
                count_of_steps=env.action_space(), batch_size=632, score_transformer_fn= env.reward_to_score_transformer())

    end_date = datetime.datetime.now()
    print('End time:', end_date)
    print('Tooked', end_date - start_date)

    torch.save(net, 'save.net')
