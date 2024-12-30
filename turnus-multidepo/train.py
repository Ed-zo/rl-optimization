import torch
import datetime


from utils.graph_generator import GraphGenerator
from utils.problem_loader import ProblemLoader
from ppo_parallel import Agent
from env import Env
from utils.utils import run_in_child_process
from model import HybridNetwork
from torch.multiprocessing import Process, Manager

def get_env(*params) -> Env:
    def startup(q):
        q.put(Env(*params))

    manager = Manager()
    q = manager.Queue()
    p = Process(target=startup, args=(q,))
    p.start()
    p.join()

    return q.get()

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    start_date = datetime.datetime.now()
    print('Start time:', start_date)

    problem_loader = ProblemLoader('data/4dep_100trips/')

    graphs = [problem_loader.load(f'RN-4-100-0{i}') for i in range(1, 21)]

    # g_generator = None
    g_generator = run_in_child_process(GraphGenerator, graphs)
    graph = None
    # graph = run_in_child_process(problem_loader.load, 'RN-4-100-01')
    env = run_in_child_process(Env, g_generator, graph)

    net = HybridNetwork(env.state_space(), env.action_space()).to(device)
    rnd_net = HybridNetwork(env.state_space(), env.action_space()).to(device)
    net.train()
    rnd_net.train()

    agent = Agent(net, rnd_net, device=device, name='4dep_100trips', ext_gamma=1, epsilon=0.2, lr=0.001, override=True)
    
    agent.training_description('Trening')

    # stop_signal_count = 0
    # def stop_signal(sig, frame):
    #     global stop_signal_count
    #     print('Stoping')
    #     if stop_signal_count > 1:
    #         exit()

    #     agent.stop_training()
    #     stop_signal_count += 1

    # signal.signal(signal.SIGINT, stop_signal)
    agent.train([g_generator, graph], Env, env.action_space(), count_of_iterations=1000, count_of_processes=2, count_of_envs=16, 
                count_of_steps=int(env.action_space() + env.MAX_VEHICLES), batch_size=1328, score_transformer_fn= env.reward_to_score_transformer())

    end_date = datetime.datetime.now()
    print('End time:', end_date)
    print('Tooked', end_date - start_date)