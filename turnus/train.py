from distutils.file_util import write_file
import signal
import torch
import torch.nn as nn
import datetime
import torch.nn.functional as F
from ppo_parallel import Agent
import torch.nn.init as init
from env import Env
import utils.graph_utils as graph_utils
from model import GCNPolicy, RNDModel


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    start_date = datetime.datetime.now()
    print('Start time:', start_date)

    problem = 2
    problem_path = f'data/{problem}'

    graph, optimal_vehicles = graph_utils.load_problem(problem_path)

    env = Env(graph, optimal_vehicles, device)

    net = GCNPolicy(env.state_space(), env.action_space()).to(device)
    # net.load_state_dict(torch.load('results/models/ppo_problem_2_gcn_4984.pt'))
    rnd_net = RNDModel(env.state_space(), env.action_space()).to(device)
    net.train()
    rnd_net.train()

    agent = Agent(net, rnd_net, device=device, name='p_2_max_vehicles', ext_gamma=1, epsilon=0.2, lr=0.001, override=True)
    
    agent.training_description('Znizenie MAX VEHICLES na 1/3')

    # stop_signal_count = 0
    # def stop_signal(sig, frame):
    #     global stop_signal_count
    #     print('Stoping')
    #     if stop_signal_count > 1:
    #         exit()

    #     agent.stop_training()
    #     stop_signal_count += 1

    # signal.signal(signal.SIGINT, stop_signal)

    agent.train([graph, optimal_vehicles], Env, graph.num_nodes, count_of_iterations=10000, count_of_processes=2, count_of_envs=16, 
                count_of_steps=env.action_space() + (env.MAX_VEHICLES * 2), batch_size=1048, score_transformer_fn= env.reward_to_score_transformer())

    end_date = datetime.datetime.now()
    print('End time:', end_date)
    print('Tooked', end_date - start_date)