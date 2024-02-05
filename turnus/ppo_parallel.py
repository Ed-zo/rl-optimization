import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from torch.multiprocessing import Process, Pipe, set_start_method
from torch_geometric.data import Batch
from utils.graph_store import GraphStore
from utils.logger import AgentLogger, ScoreLogger
from utils.utils import RunningStats, flatten_list
# import ipdb

# 1 proces
# Odohranie hry v niekolkych prostrediach
def worker(connection, env_params, env_func, count_of_iterations, count_of_envs,
           count_of_steps, ext_gamma, int_gamma, gae_lambda):
    # ipdb.set_trace()
    envs = [env_func(*env_params) for _ in range(count_of_envs)]
    # observations = GraphStore((count_of_steps, count_of_envs))
    observations, masks = list(map(list, zip(*[env.reset() for env in envs])))
    masks = torch.stack(masks)
    game_ext_rewards = np.zeros(count_of_envs)
    game_int_rewards = torch.zeros(count_of_envs)

    mem_observations = GraphStore((count_of_steps, count_of_envs))
    mem_masks = torch.zeros((count_of_steps, count_of_envs, masks.shape[1]))
    mem_log_probs = torch.zeros((count_of_steps, count_of_envs, 1))
    mem_actions = torch.zeros((count_of_steps, count_of_envs, 1), dtype=torch.long)
    mem_ext_values = torch.zeros((count_of_steps + 1, count_of_envs, 1))
    mem_ext_rewards = torch.zeros((count_of_steps, count_of_envs, 1))
    mem_int_values = torch.zeros((count_of_steps + 1, count_of_envs, 1))
    mem_int_rewards = torch.zeros((count_of_steps, count_of_envs, 1))

    for iteration in range(count_of_iterations):
        mem_non_terminals = torch.ones((count_of_steps, count_of_envs, 1))
        ext_scores = []
        int_scores = []

        # Hranie prostredia
        for step in range(count_of_steps):
            connection.send(observations)                                                                       #1 A
            logits, ext_values, int_values, int_rewards = connection.recv()                                                                  #2 B
            logits = torch.where(masks, logits, torch.tensor(-1e+8))
            probs = F.softmax(logits, dim=-1)
            actions = probs.multinomial(num_samples=1)
            log_probs = F.log_softmax(logits, dim=-1).gather(1, actions)

            mem_observations[step] = observations
            mem_masks[step] = masks
            mem_log_probs[step] = log_probs
            mem_actions[step] = actions
            mem_ext_values[step] = ext_values
            mem_int_values[step] = int_values
            mem_int_rewards[step] = int_rewards
            
            game_int_rewards += int_rewards.squeeze(-1)

            # Vykonanie jedneho kroku v kazdom prostredi v ramci workera
            for idx in range(count_of_envs):
                observation, mask, ext_reward, terminal, _ = envs[idx].step(actions[idx, 0].item())
                mem_ext_rewards[step, idx, 0] = ext_reward
                game_ext_rewards[idx] += ext_reward
                # if ext_reward < 0:
                    # mem_non_terminals[step, idx, 0] = 0

                if terminal:
                    mem_non_terminals[step, idx, 0] = 0
                    ext_scores.append(game_ext_rewards[idx])
                    int_scores.append(game_int_rewards[idx].item())
                    game_ext_rewards[idx] = 0
                    game_int_rewards[idx] = 0
                    observation, mask = envs[idx].reset()
                observations[idx] = observation
                masks[idx] = mask

        connection.send(observations)                                                                           #3 A
        mem_ext_values[step + 1], mem_int_values[step + 1] = connection.recv()                                                                 #4 B
        

        '''
            Extrinsic values - calculating advantage using gae
        '''
        mem_ext_rewards = torch.clamp(mem_ext_rewards, -1.0, 1.0)
        ext_advantages = torch.zeros((count_of_steps, count_of_envs, 1))
        ext_values = torch.zeros((count_of_steps, count_of_envs, 1))
        ext_gae = torch.zeros((count_of_envs, 1))

        for step in reversed(range(count_of_steps)):
            delta = mem_ext_rewards[step] + ext_gamma * mem_ext_values[step + 1] * mem_non_terminals[step] \
                    - mem_ext_values[step]
            ext_gae = delta + ext_gamma * gae_lambda * ext_gae * mem_non_terminals[step]
            ext_values[step] = ext_gae + mem_ext_values[step]
            ext_advantages[step] = ext_gae.clone()

        '''
            Intrinsic values - calculating advantage using gae
        '''
        mem_int_rewards = torch.clamp(mem_int_rewards, 0., 1.)
        int_advantages = torch.zeros((count_of_steps, count_of_envs, 1))
        int_values = torch.zeros((count_of_steps, count_of_envs, 1))
        int_gae = torch.zeros((count_of_envs, 1))

        for step in reversed(range(count_of_steps)):
            delta = mem_int_rewards[step] + int_gamma * mem_int_values[step + 1] * mem_non_terminals[step] \
                    - mem_int_values[step]
            int_gae = delta + int_gamma * gae_lambda * int_gae * mem_non_terminals[step]
            int_values[step] = int_gae + mem_int_values[step]
            int_advantages[step] = int_gae.clone()

        connection.send([mem_observations, mem_masks, mem_log_probs, mem_actions, ext_values, int_values, ext_advantages, int_advantages, ext_scores, int_scores])                                #5 A
    connection.recv()
    connection.close()


class Agent:
    def __init__(self, model, rnd_model, ext_gamma=0.997, int_gamma=0.99, epsilon=0.1,
                 coef_value=0.5, coef_entropy=0.001, gae_lambda=0.95,
                 name='ppo', path='results/', device='cpu', lr = 0.00025):

        self.model = model
        self.rnd_model = rnd_model
        self.model.to(device)
        self.rnd_model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.rnd_optimizer = torch.optim.Adam(self.rnd_model.parameters(), lr = lr)

        self.ext_gamma = ext_gamma
        self.int_gamma = int_gamma
        self.coef_value = coef_value
        self.coef_entropy = coef_entropy
        self.gae_lambda = gae_lambda

        self.lower_bound = 1 - epsilon
        self.upper_bound = 1 + epsilon

        self.name = name
        self.path = path
        self.device = device

    def train(self, env_params, env_func, count_of_actions,
              count_of_iterations=10000, count_of_processes=2,
              count_of_envs=16, count_of_steps=128, count_of_epochs=4,
              batch_size=512, score_transformer_fn = None):

        print('Training is starting')

        loss_logger = AgentLogger(f'{self.path}/data/{self.name}_loss.csv', ['avg_score', 'policy', 'ext_value', 'int_value', 'entropy', 'rnd', 'lr'])
        score_logger = ScoreLogger(f'{self.path}/data/{self.name}.csv', score_transformer_fn=score_transformer_fn)

        lr_scheduler = LinearLR(self.optimizer, start_factor=1, end_factor=0.0001, total_iters=int(count_of_iterations / 2))
        buffer_size = count_of_processes * count_of_envs * count_of_steps
        batches_per_iteration = count_of_epochs * buffer_size / batch_size

        processes, connections = [], []
        for _ in range(count_of_processes):
            parr_connection, child_connection = Pipe()
            process = Process(target=worker, args=(
                child_connection, env_params, env_func, count_of_iterations,
                count_of_envs, count_of_steps, self.ext_gamma, self.int_gamma, self.gae_lambda))
            connections.append(parr_connection)
            processes.append(process)
            process.start()

        for iteration in range(count_of_iterations):
            for step in range(count_of_steps):
                observations = [conn.recv() for conn in connections]                                #1 B

                with torch.no_grad():
                    observations = flatten_list(observations)
                    observations = Batch.from_data_list(observations).to(self.device)
                    logits, ext_values, int_values = self.model(observations)

                    '''
                    RND - intrinsic rewards
                    '''
                    # rnd_pred, rnd_targ = self.rnd_model(observations)

                    # int_rewards = (rnd_pred - rnd_targ) ** 2
                    # int_rewards = int_rewards.sum(dim=1) / 2
                    int_rewards = torch.zeros((count_of_processes * count_of_envs))

                # If you selected actions in the main process, your iteration
                # would last about 0.5 seconds longer (measured on 2 processes)
                logits = logits.view(-1, count_of_envs, count_of_actions).cpu()
                ext_values = ext_values.view(-1, count_of_envs, 1).cpu()
                int_values = int_values.view(-1, count_of_envs, 1).cpu()
                int_rewards = int_rewards.view(-1, count_of_envs, 1).cpu()

                for idx in range(count_of_processes):
                    connections[idx].send((logits[idx], ext_values[idx], int_values[idx], int_rewards[idx]))       #2 A

            observations = [conn.recv() for conn in connections]                                                  #3 B
            observations = flatten_list(observations)
            observations = Batch.from_data_list(observations).to(self.device)

            with torch.no_grad():
                _, ext_values, int_values = self.model(observations)
                ext_values = ext_values.view(-1, count_of_envs, 1).cpu()
                int_values = int_values.view(-1, count_of_envs, 1).cpu()

            for conn_idx in range(count_of_processes):
                connections[conn_idx].send((ext_values[conn_idx], int_values[conn_idx]))                                        #4 A

            mem_observations, mem_masks, mem_log_probs, mem_actions, \
                mem_target_ext_values, mem_target_int_values, mem_ext_advantages, mem_int_advantages, end_games = [], [], [], [], [], [], [], [], []

            for connection in connections:
                observations, masks, log_probs, actions, target_ext_values, target_int_values, ext_advantages, int_advantages, score_of_end_games, int_scores_of_end_games = connection.recv()     #5 B
                
                with open('debug.csv', 'a+') as f:
                    f.write(f'{iteration}, "{json.dumps(score_of_end_games)}", "{json.dumps(int_scores_of_end_games)}"\n')

                mem_observations.extend(observations.flatten())
                mem_masks.append(masks)
                mem_actions.append(actions)
                mem_log_probs.append(log_probs)
                mem_target_ext_values.append(target_ext_values)
                mem_target_int_values.append(target_int_values)
                mem_ext_advantages.append(ext_advantages)
                mem_int_advantages.append(int_advantages)
                end_games.extend(score_of_end_games)

            episode, avg_score, better_score = score_logger.log(iteration, end_games)

            mem_observations = Batch.from_data_list(mem_observations).to(self.device)
            mem_masks = torch.stack(mem_masks).to(self.device).bool().view(-1, count_of_actions)
            mem_actions = torch.stack(mem_actions).to(self.device).view(-1, 1)
            mem_log_probs = torch.stack(mem_log_probs).to(self.device).view(-1, 1)
            mem_target_ext_values = torch.stack(mem_target_ext_values).to(self.device).view(-1, 1)
            mem_target_int_values = torch.stack(mem_target_int_values).to(self.device).view(-1, 1)
            mem_ext_advantages = torch.stack(mem_ext_advantages).to(self.device).view(-1, 1)
            mem_int_advantages = torch.stack(mem_int_advantages).to(self.device).view(-1, 1)
            mem_advantages = 2 * mem_ext_advantages + mem_int_advantages
            mem_advantages = (mem_advantages - torch.mean(mem_advantages)) / (torch.std(mem_advantages) + 1e-5)

            s_policy, s_ext_value, s_int_value, s_entropy, s_rnd = 0, 0, 0, 0, 0

            for epoch in range(count_of_epochs):
                perm = torch.randperm(buffer_size, device=self.device).view(-1, batch_size)
                for idx in perm:
                    obs = Batch.from_data_list(mem_observations[idx])
                    logits, ext_values, int_values = self.model(obs)
                    logits = torch.where(mem_masks[idx], logits, torch.tensor(-1e+8, device=self.device))
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs.gather(1, mem_actions[idx])

                    entropy_loss = (log_probs * probs).sum(1, keepdim=True).mean()
                    ext_value_loss = F.mse_loss(ext_values, mem_target_ext_values[idx])
                    int_value_loss = F.mse_loss(int_values, mem_target_int_values[idx])

                    rnd_pred, rnd_targ = self.rnd_model(obs)
                    loss_rnd = (rnd_targ - rnd_pred)**2

                    #random loss regularisation, 25% non zero for 128envs, 100% non zero for 32envs
                    prob            = 16.0/(count_of_envs * count_of_processes)
                    random_mask     = torch.rand(loss_rnd.shape).to(loss_rnd.device)
                    random_mask     = 1.0*(random_mask < prob)
                    loss_rnd        = (loss_rnd*random_mask).sum() / (random_mask.sum() + 0.00000001)

                    ratio = torch.exp(new_log_probs - mem_log_probs[idx])
                    surr_policy = ratio * mem_advantages[idx]
                    surr_clip = torch.clamp(ratio, self.lower_bound, self.upper_bound) \
                                * mem_advantages[idx]
                    policy_loss = - torch.min(surr_policy, surr_clip).mean()

                    s_policy += policy_loss.item()
                    s_ext_value += ext_value_loss.item()
                    s_int_value += int_value_loss.item()
                    s_entropy += entropy_loss.item()
                    s_rnd += loss_rnd.item()

                    self.optimizer.zero_grad()
                    loss = policy_loss + self.coef_value * (ext_value_loss + int_value_loss) \
                        + self.coef_entropy * entropy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

                    self.rnd_optimizer.zero_grad()
                    loss_rnd.backward()
                    # torch.nn.utils.clip_grad_norm_(self.rnd_model.parameters(), 0.5)
                    self.rnd_optimizer.step()

            lr_scheduler.step()

            loss_logger.log(iteration, episode, avg_score, 
                            s_policy / batches_per_iteration, 
                            s_ext_value / batches_per_iteration,
                            s_int_value / batches_per_iteration,
                            s_entropy / batches_per_iteration,
                            s_rnd / batches_per_iteration,
                            self.optimizer.param_groups[0]['lr'])

            if better_score:
                self.save_model(iteration)

        print('Training has ended, best avg score is ', score_logger.mva.get_best_avg_score())

        for connection in connections:
            connection.send(1)
        for process in processes:
            process.join()

    def save_model(self, iteration):
        os.makedirs(f'{self.path}models/', exist_ok=True)
        torch.save(self.model.state_dict(), f'{self.path}models/{self.name}_{str(iteration)}.pt')


    def load_model(self, path):
        print('Loading model from', path)
        self.model.load_state_dict(torch.load(path))