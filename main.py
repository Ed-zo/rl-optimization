import torch
import torch.nn as nn
import datetime
import torch.nn.functional as F
from env_parallel import Env
from ppo import Agent
import torch.nn.init as init
import numpy as np
import signal


# def weights_init_xavier(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1 or classname.find('Linear') != -1:
#         init.xavier_uniform_(m.weight)


class PolicyValueModel(nn.Module):
    def __init__(self, count_of_candidates):
        super(PolicyValueModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)

        self.apool = nn.AdaptiveAvgPool2d(1)

        self.size = 32

        self.fc_p1 = nn.Linear(self.size, 32)
        self.fc_p2 = nn.Linear(32, count_of_candidates)

        self.fc_v1 = nn.Linear(self.size, 32)
        self.fc_v2 = nn.Linear(32, 1)

        features_layers = [self.conv1, self.conv2]
        for layer in features_layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        output_layers = [self.fc_p1, self.fc_p2, self.fc_v1, self.fc_v2]
        for layer in output_layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))

        x = self.apool(x)

        x = torch.flatten(x, 1)

        x_logit = F.relu(self.fc_p1(x))
        logit = self.fc_p2(x_logit)

        x_value = F.relu(self.fc_v1(x))
        value = self.fc_v2(x_value)

        return logit, value


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    start_date = datetime.datetime.now()

    env_candidates = 5
    env_p = 4
    data = 'data/test-5'
    env_count = 40
    results_path = 'results/'

    env = Env(env_p, env_candidates, env_count, data, device)

    net = PolicyValueModel(env_candidates)
    # net = torch.load('models/save.net')

    agent = Agent(net, device=device, lr=0.01, name='p_med', results_path=results_path, epsilon=0.2, td_steps=env_p,)
    signal.signal(signal.SIGINT, agent.stop_training)

    agent.train(env=env, count_of_envs=env_count, input_dim=(3, env_candidates, env_candidates),
                count_of_iterations=40, count_of_steps=256, batch_size=256)
    
    # agent.test(env)
    
    print(datetime.datetime.now() - start_date)

    # torch.save(net, 'models/save.net')
