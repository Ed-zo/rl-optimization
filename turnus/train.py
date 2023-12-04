import inspect
import torch
import torch.nn as nn
import datetime
import torch.nn.functional as F
from ppo_parallel import Agent
import torch.nn.init as init
import numpy as np
import signal
from env import Env
import graph_utils
from model import GCN

# print(inspect.getfile(GCN))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    start_date = datetime.datetime.now()
    print('Start time:', start_date)
    graph = graph_utils.load_graph('data/1')

    env = Env(graph, device)

    net = GCN(env.state_space(), env.action_space()).to(device)

    agent = Agent(net, 'Adam', device=device, name='ppo_2', path='results/', gamma=1, epsilon=0.1, lr=0.0005)

    agent.train([graph, device], Env, graph.num_nodes, count_of_iterations=10000, count_of_processes=4, count_of_envs=8, count_of_steps=52*2, batch_size=832)

    end_date = datetime.datetime.now()
    print('End time:', end_date)
    print('Tooked', end_date - start_date)

    torch.save(net, 'save.net')
