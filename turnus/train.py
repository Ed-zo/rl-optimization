import torch
import datetime
import torch.nn.functional as F
from ppo_parallel import Agent
# from env_local_rewards import EnvLocalRewards as Env
from env import Env
from utils.graph_generator import GraphGenerator
import utils.graph_utils as graph_utils
from model import GCNPolicy, RNDModel
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    start_date = datetime.datetime.now()
    print('Start time:', start_date)

    g_generator = GraphGenerator(problems=[7, 8, 9, 10], size=51)

    env = get_env(g_generator)

    net = GCNPolicy(env.state_space(), env.action_space()).to(device)
    # net.load_state_dict(torch.load('results/240214#p_2_manualne_ladenie_lr/models/iter_2006.pt'))
    rnd_net = RNDModel(env.state_space(), env.action_space()).to(device)
    net.train()
    rnd_net.train()

    agent = Agent(net, rnd_net, device=device, name='p_rand_51', ext_gamma=1, epsilon=0.2, lr=0.001, override=True)
    
    agent.training_description('Nahodne samplovanie grafov o velkosti 51')

    # stop_signal_count = 0
    # def stop_signal(sig, frame):
    #     global stop_signal_count
    #     print('Stoping')
    #     if stop_signal_count > 1:
    #         exit()

    #     agent.stop_training()
    #     stop_signal_count += 1

    # signal.signal(signal.SIGINT, stop_signal)

    agent.train([g_generator], Env, env.action_space(), count_of_iterations=10000, count_of_processes=2, count_of_envs=32, 
                count_of_steps=env.action_space() + env.MAX_VEHICLES, batch_size=1632, score_transformer_fn= env.reward_to_score_transformer())

    end_date = datetime.datetime.now()
    print('End time:', end_date)
    print('Tooked', end_date - start_date)