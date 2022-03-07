import torch
import torch.nn as nn
import datetime
import torch.nn.functional as F
from dqn import AgentDDQN
from env_parallel import Env
from experience_replay import ExperienceReplay
import numpy as np
import signal

# def weights_init_xavier(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1 or classname.find('Linear') != -1:
#         init.xavier_uniform_(m.weight)


class PolicyValueModel(nn.Module):
    def __init__(self, count_of_actions):
        super(PolicyValueModel, self).__init__()

        self.fc1 = nn.Linear(count_of_actions, 8)
        self.fc2 = nn.Linear(8, count_of_actions)

        output_layers = [self.fc1, self.fc2]
        for layer in output_layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x =self.fc2(x)

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    start_date = datetime.datetime.now()

    env_candidates = 5
    env_p = 4
    env_count = 1
    results_path = 'results/'

    env = Env(env_p, env_candidates, env_count, 'data/test-5', device)
    net = PolicyValueModel(env_candidates)
    experience_replay = ExperienceReplay(10000, (env_candidates, ))

    agent = AgentDDQN(0.99, env_candidates, net, experience_replay, 0.001, device, epsilon_dec=0.0002)
    
    scores = []
    d_now = datetime.datetime.now()
    for i in range(1, 2000):
        score = 0
        terminal = False
        state = env.reset()

        while not terminal:
            action = agent.choose_action(state)
            state_, reward, terminal = env.step(action)
            agent.store(state[0], action, reward[0], state_[0], terminal)
            agent.learn()
            state = state_
            score += reward[0]

        scores.append(score)

        print('episode: ', i, '\t\tscore: ', + score, '\t\taverage score:',
              np.average(scores[-100:]), '\t\tepsilon: ', agent.epsilon)
        
    
    print(datetime.datetime.now() - start_date)