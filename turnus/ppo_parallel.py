import numpy as np
import torch
import torch.nn.functional as F
from torch.multiprocessing import Process, Pipe
from torch_geometric.data import Batch
from graph_store import GraphStore
from utils import MovingAverageScore, write_to_file
import ipdb


def worker(connection, env_params, env_func, count_of_iterations, count_of_envs,
           count_of_steps, gamma, gae_lambda):
    # ipdb.set_trace()
    envs = [env_func(*env_params) for _ in range(count_of_envs)]
    # observations = GraphStore((count_of_steps, count_of_envs))
    observations, masks = list(map(list, zip(*[env.reset() for env in envs])))
    game_score = np.zeros(count_of_envs)

    mem_log_probs = torch.zeros((count_of_steps, count_of_envs, 1))
    mem_actions = torch.zeros((count_of_steps, count_of_envs, 1), dtype=torch.long)
    mem_values = torch.zeros((count_of_steps + 1, count_of_envs, 1))
    mem_rewards = torch.zeros((count_of_steps, count_of_envs, 1))
    mem_masks = torch.zeros((count_of_steps, count_of_envs, masks[0].shape[0]))

    for iteration in range(count_of_iterations):
        mem_non_terminals = torch.ones((count_of_steps, count_of_envs, 1))
        scores = []

        # Hranie prostredia
        for step in range(count_of_steps):
            connection.send((observations, masks))                                                              #1 A
            logits, values = connection.recv()                                                                  #2 B
            logits = torch.where(masks, logits, torch.tensor(-1e+8))
            probs = F.softmax(logits, dim=-1)
            actions = probs.multinomial(num_samples=1)
            log_probs = F.log_softmax(logits, dim=-1).gather(1, actions)

            mem_log_probs[step] = log_probs
            mem_actions[step] = actions
            mem_values[step] = values

            # Vykonanie jedneho kroku v kazdom prostredi v ramci workera
            for idx in range(count_of_envs):
                observation, mask, reward, terminal, _ = envs[idx].step(actions[idx, 0].item())
                mem_rewards[step, idx, 0] = reward
                game_score[idx] += reward
                if reward < 0:
                    mem_non_terminals[step, idx, 0] = 0

                if terminal:
                    mem_non_terminals[step, idx, 0] = 0
                    scores.append(game_score[idx])
                    game_score[idx] = 0
                    observation, mask = envs[idx].reset()
                observations[idx] = observation
                masks[idx] = mask

        connection.send((observations, masks))                                                                   #3 A
        mem_values[step + 1] = connection.recv()                                                                 #4 B
        mem_rewards = torch.clamp(mem_rewards, -1.0, 1.0)
        advantages = torch.zeros((count_of_steps, count_of_envs, 1))
        values = torch.zeros((count_of_steps, count_of_envs, 1))
        t_gae = torch.zeros((count_of_envs, 1))

        for step in reversed(range(count_of_steps)):
            delta = mem_rewards[step] + gamma * mem_values[step + 1] * mem_non_terminals[step] \
                    - mem_values[step]
            t_gae = delta + gamma * gae_lambda * t_gae * mem_non_terminals[step]
            values[step] = t_gae + mem_values[step]
            advantages[step] = t_gae.clone()

        connection.send([mem_log_probs, mem_actions, values, advantages, scores])                                #5 A
    connection.recv()
    connection.close()


class Agent:
    def __init__(self, model, optimizer, gamma=0.997, epsilon=0.1,
                 coef_value=0.5, coef_entropy=0.001, gae_lambda=0.95,
                 name='ppo', path='results/', device='cpu', lr = 0.00025):

        self.model = model
        self.model.to(device)

        if optimizer == 'Adam':
            print('optimizer: Adam')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        elif optimizer == 'SGD':
            print('optimizer: SGD wiht momentum = 0.9')
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, momentum = 0.9)
        elif optimizer == 'RMS':
            print('optimizer: RMSProp')
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = lr)
        else:
            print('optimizer: SGD wiht momentum = 0.0')
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr)

        self.gamma = gamma
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
              batch_size=512, input_dim=(4, 96, 96)):

        print('Training is starting')

        logs_score = 'iteration,episode,avg_score,best_avg_score,best_score'
        logs_loss = 'iteration,episode,policy,value,entropy'

        score = MovingAverageScore()
        buffer_size = count_of_processes * count_of_envs * count_of_steps
        batches_per_iteration = count_of_epochs * buffer_size / batch_size

        processes, connections = [], []
        for _ in range(count_of_processes):
            parr_connection, child_connection = Pipe()
            process = Process(target=worker, args=(
                child_connection, env_params, env_func, count_of_iterations,
                count_of_envs, count_of_steps, self.gamma, self.gae_lambda))
            connections.append(parr_connection)
            processes.append(process)
            process.start()

        mem_dim = (count_of_processes, count_of_steps, count_of_envs)
        # mem_observations = torch.zeros((mem_dim + input_dim), device=self.device)
        mem_observations = GraphStore(mem_dim)
        mem_actions = torch.zeros((*mem_dim, 1), device=self.device, dtype=torch.long)
        mem_log_probs = torch.zeros((*mem_dim, 1), device=self.device)
        mem_values = torch.zeros((*mem_dim, 1), device=self.device)
        mem_advantages = torch.zeros((*mem_dim, 1), device=self.device)
        # mem_masks = torch.zeros((count_of_steps, count_of_envs, masks[0].shape[0]))

        for iteration in range(count_of_iterations):
            for step in range(count_of_steps):
                obs_msk = [conn.recv() for conn in connections]                                     #1 B
                observations, masks = list(map(list, zip(*obs_msk)))
                # masks = torch.stack(masks).to(self.device)
                mem_observations[:, step] = observations

                with torch.no_grad():
                    observations = Batch.from_data_list([*observations[0], *observations[1]]).to(self.device)
                    logits, values = self.model(observations)

                # If you selected actions in the main process, your iteration
                # would last about 0.5 seconds longer (measured on 2 processes)
                logits = logits.view(-1, count_of_envs, count_of_actions).cpu()
                values = values.view(-1, count_of_envs, 1).cpu()

                for idx in range(count_of_processes):
                    connections[idx].send((logits[idx], values[idx]))                               #2 A

            observations = [conn.recv() for conn in connections]                                    #3 B
            Batch.from_data_list(observations).to(self.device)
            # observations = torch.stack(observations).to(self.device)

            with torch.no_grad():
                _, values = self.model(torch.stack(observations).to(self.device).view(-1, *input_dim))
                values = values.view(-1, count_of_envs, 1).cpu()

            for conn_idx in range(count_of_processes):
                connections[conn_idx].send(values[conn_idx])                                        #4 A

            mem_observations, mem_actions, mem_log_probs, mem_target_values, mem_advantages, end_games = [], [], [], [], [], []

            for connection in connections:
                observations, actions, log_probs, target_values, advantages, score_of_end_games = connection.recv()     #5 B
                
                mem_observations.append(observations)
                mem_actions.append(actions)
                mem_log_probs.append(log_probs)
                mem_target_values.append(target_values)
                mem_advantages.append(advantages)
                end_games.extend(score_of_end_games)

            count_of_end_games = len(end_games)
            if count_of_end_games > 0:
                new_score = True
                count_of_episodes += count_of_end_games
                scores.extend(end_games)
                best_score = max(best_score, np.max(end_games))

                length = len(scores)
                if length > 100:
                    scores = scores[length - 100:]
                avg_score = np.average(scores)
                prev_avg_score = best_avg_score
                best_avg_score = max(best_avg_score, avg_score)
                logs += '\n' + str(iteration) + ',' + str(count_of_episodes) + ',' + str(avg_score) + ',' + str(best_score) + ',' + str(best_avg_score)
                if iteration % 1 == 0:
                    print('iteration', iteration, '\tepisode', count_of_episodes, '\tavg score', avg_score, '\tbest score', best_score, '\tbest avg score', best_avg_score)

            # mem_observations = torch.stack(mem_observations).to(self.device).view((-1, ) + input_dim)
            mem_observations = mem_observations.to(self.device)
            mem_actions = torch.stack(mem_actions).to(self.device).view(-1, 1)
            mem_log_probs = torch.stack(mem_log_probs).to(self.device).view(-1, 1)
            mem_target_values = torch.stack(mem_target_values).to(self.device).view(-1, 1)
            mem_advantages = torch.stack(mem_advantages).to(self.device).view(-1, 1)
            mem_advantages = (mem_advantages - torch.mean(mem_advantages)) / (torch.std(mem_advantages) + 1e-5)

            sum_policy_loss, sum_value_loss, sum_entropy_loss = 0, 0, 0

            for epoch in range(count_of_epochs):
                perm = torch.randperm(buffer_size, device=self.device).view(-1, batch_size)
                for idx in perm:
                    logits, values = self.model(mem_observations[idx])
                    probs = F.softmax(logits, dim=-1)
                    log_probs = F.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs.gather(1, mem_actions[idx])

                    entropy_loss = (log_probs * probs).sum(1, keepdim=True).mean()
                    value_loss = F.mse_loss(values, mem_values[idx])

                    ratio = torch.exp(new_log_probs - mem_log_probs[idx])
                    surr_policy = ratio * mem_advantages[idx]
                    surr_clip = torch.clamp(ratio, self.lower_bound, self.upper_bound) \
                                * mem_advantages[idx]
                    policy_loss = - torch.min(surr_policy, surr_clip).mean()

                    s_policy += policy_loss.item()
                    s_value += value_loss.item()
                    s_entropy += entropy_loss.item()

                    self.optimizer.zero_grad()
                    loss = policy_loss + self.coef_value * value_loss \
                           + self.coef_entropy * entropy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()


            logs_losses +=  '\n' + str(iteration) + ',' + str(len(scores)) + ',' + str(sum_policy_loss / count_of_losses) + ',' + str(sum_value_loss / count_of_losses) + ',' + str(sum_entropy_loss / count_of_losses)
            if iteration % 1 == 0:
                write_to_file(logs, self.results_path + 'data/' + self.name + '.txt')
                write_to_file(logs_losses, self.results_path + 'data/' + self.name + '_loss.txt')
                if best_avg_score > prev_avg_score:
                    self.save_model()



            logs_loss += '\n' + str(iteration) + ',' \
                         + str(avg_score) + ',' \
                         + str(s_policy / batches_per_iteration) + ',' \
                         + str(s_value / batches_per_iteration) + ',' \
                         + str(s_entropy / batches_per_iteration)

            if iteration % 10 == 0:
                write_to_file(logs_score, self.path + 'data/' + self.name + '.txt')
                write_to_file(logs_loss, self.path + 'data/' + self.name + '_loss.txt')
        print('Training has ended, best avg score is ', score.get_best_avg_score())

        for connection in connections:
            connection.send(1)
        for process in processes:
            process.join()

    # def save_model(self):
    #     torch.save(self.model.state_dict(), self.results_path + 'models/' + self.name + str(self.iteration) + '_ppo.pt')


    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))