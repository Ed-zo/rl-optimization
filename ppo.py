from asyncore import write
import torch
import torch.nn.functional as F
import numpy as np
from util import write_to_file

class Agent:
    def __init__(self, model, gamma = 0.99, entropy_loss_coef = 0.01, value_loss_coef = 0.5, epsilon = 0.1, lr = 0.001, lr_decay = 0.5, lr_steps = 500,
                 name = 'ppo', optim = 'Adam', weight_decay = 0.0001, device = 'cpu', td_steps = 10, results_path = 'results/'):
        self.device = device
        self.model = model
        model.to(device)

        if optim == 'Adam':
            print('optimizer: Adam')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        elif optim == 'SGD':
            print('optimizer: SGD wiht momentum = 0.9')
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, momentum = 0.9, weight_decay = weight_decay)
        elif optim == 'RMS':
            print('optimizer: RMSProp')
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        else:
            print('optimizer: SGD wiht momentum = 0.0')
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, weight_decay = weight_decay)

        self.gamma = gamma
        self.entropy_loss_coef = entropy_loss_coef
        self.value_loss_coef = value_loss_coef

        self.upper_bound = 1 + epsilon
        self.lower_bound = 1 - epsilon

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_steps = lr_steps
        self.td_steps = td_steps

        self.name = name
        self.results_path = results_path
        
        self.gae_lambda = 0.95

    def test(self, env):
        terminal = False
        observation = env.reset()
        rewards = 0
        while not terminal:
            with torch.no_grad():
                logits, values = self.model(observation.to(self.device))

            probs = F.softmax(logits, dim = -1)

            action_mask = (1 - observation).abs().reshape(probs.shape)
            probs = (probs * action_mask) / probs.sum(dim=-1).reshape((probs.shape[0], 1))
            actions = probs.argmax(dim = -1)
            observation, reward, terminal = env.step(actions)
            rewards += reward

        print('Obj function: {0} State: {1} Rewards: {2}'.format(env.compute_objective_function()[0], list(map(int, observation[0].tolist())), rewards.item()))
           

    def train(self, env, input_dim, count_of_envs, count_of_iterations, count_of_steps, batch_size, count_of_epochs = 4, first_iteration = 0, seed = None):
        print('Training is starting')
        count_of_steps_per_iteration = count_of_steps * count_of_envs
        mse = torch.nn.MSELoss()
        logs = 'iteration,episode,score,avg_score,best_score,best_avg_score,obj_fun,best_obj'
        logs_losses = 'iteration,episode,policy,value,entropy'

        if seed is None:
            seed = [i for i in range(count_of_envs)]

        observations = env.reset()

        scores, curr_scores = [], torch.zeros(count_of_envs, device = self.device)
        best_avg_score, best_score, best_obj, best_observation = -1e9, -1e9, 1e9, {}

        for iteration in range(first_iteration, count_of_iterations):
            if iteration > 0 and iteration % self.lr_steps == 0:
                #self.lr *= self.lr_decay
                #for g in self.optimizer.param_groups:
                #    g['lr'] = self.lr
                torch.save(self.model.state_dict(), 'models/' + self.name + '_' + str(iteration) + '.pt')

            mem_pred_values = torch.zeros((count_of_steps + 1, count_of_envs, 1))
            mem_log_probs = torch.zeros((count_of_steps, count_of_envs, 1))
            mem_rewards = torch.zeros((count_of_steps, count_of_envs, 1))
            mem_non_terminals = torch.ones((count_of_steps, count_of_envs, 1))
            mem_actions = torch.zeros((count_of_steps, count_of_envs, 1)).long()
            mem_observations = torch.zeros((count_of_steps, count_of_envs, ) + input_dim)

            for step in range(count_of_steps):
                with torch.no_grad():
                    logits, values = self.model(observations.to(self.device))

                mem_observations[step] = observations.clone()
                probs, log_probs = F.softmax(logits, dim = -1), F.log_softmax(logits, dim = -1)

                #==== MASKING =====
                # action_mask = (observations - 1) * -1
                # actions = action_mask.multinomial(num_samples=1).detach().cpu()
                action_mask = (1 - observations).abs().reshape(probs.shape)
                probs = (probs * action_mask) / probs.sum(dim=-1).reshape((probs.shape[0], 1))
                log_probs = (log_probs * action_mask) / log_probs.sum(dim=-1).reshape((log_probs.shape[0], 1))
                
                actions = probs.multinomial(num_samples=1).detach()
                #==== MASKING =====

                mem_actions[step] = actions.cpu()
                mem_log_probs[step] = log_probs.cpu().gather(1, mem_actions[step])
                mem_pred_values[step] = values.cpu()

                #==== Paralel env =====
                observations, rewards, terminal = env.step(actions)
                mem_rewards[step, :, 0] = rewards
                curr_scores += rewards

                if terminal:
                    mem_non_terminals[step] = 0
                    obj = env.compute_objective_function().tolist()

                    # logging
                    curr_scores_list = curr_scores.view(-1).tolist()
                    curr_scores[:] = 0
                    scores.extend(curr_scores_list)

                    max_score = max(curr_scores_list)
                    if(best_score < max_score):
                        indices = np.argwhere(curr_scores_list == np.amax(curr_scores_list)).flatten()
                        # Magic
                        mem_rewards[step, indices, 0] = 1
                        torch.save(self.model, 'models/' + self.name + '_best.pt')
                        best_score = max_score
                        best_observation = { 'observation': observations[indices[0]], 'iteration': iteration, 'episode': len(scores) }

                    observations = env.reset()
                    avg_score = np.average(scores[-100:])

                    #save best model
                    if(best_avg_score >= avg_score):
                        torch.save(self.model, 'models/' + self.name + '_avgbest.pt')

                    best_avg_score = max(best_avg_score, avg_score)
                    best_obj = min(min(obj), best_obj)
                    episode = len(scores)
                    curr_score = curr_scores_list[0]

                    if episode % 50 == 0:
                        print('iteration: ', '{:7d}'.format(iteration), '\tepsiode: ', episode, '\tscore: ', '{:06.4f}'.format(curr_score), '\tavg score: ', 
                            '{:06.4f}'.format(avg_score), '\tbest score: ', '{:06.4f}'.format(best_score), 
                            '\tbest avg score: ', '{:06.4f}'.format(best_avg_score), '\tBest obj:', best_obj)

                    logs += '\n' + str(iteration) + ',' + str(episode) + ',' + str(curr_score) + ',' + str(avg_score) + ',' + str(best_score) + ',' + str(best_avg_score) + ',' + str()
                    if episode % count_of_envs == 0:
                        write_to_file(logs, self.results_path + self.name + '.csv')
                #==== Paralel env =====                       
                        

            advantages = torch.zeros((count_of_steps, count_of_envs, 1))
            target_values = torch.zeros((count_of_steps, count_of_envs, 1))

            with torch.no_grad():
                _, R = self.model(observations.to(self.device))
                mem_pred_values[count_of_steps] = R.cpu()
            
            mem_rewards = torch.clamp(mem_rewards, -1.0, 1.0)
            '''
            last_gae = 0
            for step in reversed(range(count_of_steps)): 
                for j in range(10):
                    
                #delta = mem_rewards[step] + self.gamma * mem_pred_values[step+1] * mem_non_terminals[step] - mem_pred_values[step]
                #last_gae = delta + self.gamma * self.gae_lambda * last_gae * mem_non_terminals[step]
                advantages[step] = #last_gae 
                target_values[step] = #last_gae + delta
            '''
            
            for step in range(count_of_steps):
                interval = min(step + self.td_steps, count_of_steps)
                R = mem_pred_values[interval]
                for i in reversed(range(step, interval)):
                    R = mem_rewards[i] + self.gamma * R * mem_non_terminals[i]
                target_values[step] = R.detach()
                advantages[step] = (R - mem_pred_values[step]).detach()

            mem_observations = mem_observations.view((-1,) + input_dim)
            mem_actions = mem_actions.to(self.device).view(-1, 1)
            mem_log_probs = mem_log_probs.to(self.device).view(-1, 1)
            target_values = target_values.to(self.device).view(-1, 1)
            advantages = advantages.to(self.device).view(-1, 1)
            advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-5)

            loss_list = [[], [], []]
            for epoch in range(count_of_epochs):
                perm = torch.randperm(count_of_steps_per_iteration)
                for batch in range(0, count_of_steps_per_iteration, batch_size):
                    indices = perm[batch:batch+batch_size]

                    logits, values = self.model(mem_observations[indices].to(self.device))
                    probs, log_probs = F.softmax(logits, dim=-1), F.log_softmax(logits, dim=-1)
                    new_log_probs = log_probs.gather(1, mem_actions[indices])
                    entropy_loss = (log_probs * probs).sum(1, keepdim=True).mean()
                    value_loss = mse(values, target_values[indices])

                    adv = advantages[indices] 
                    ratio = torch.exp(new_log_probs - mem_log_probs[indices])
                    surr_policy = ratio * adv
                    surr_clip = torch.clamp(ratio, self.lower_bound, self.upper_bound) * adv

                    policy_loss = - torch.min(surr_policy, surr_clip).mean()

                    loss_list[0].append(policy_loss.detach().item())
                    loss_list[1].append(value_loss.detach().item())
                    loss_list[2].append(entropy_loss.detach().item())

                    self.optimizer.zero_grad()
                    loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_loss_coef * entropy_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

            logs_losses +=  '\n' + str(iteration) + ',' + str(len(scores)) + ',' + str(np.average(loss_list[0])) + ',' + str(np.average(loss_list[1])) + ',' + str(np.average(loss_list[2]))
            if iteration % 10 == 0:
                write_to_file(logs_losses, self.results_path + self.name + '_loss.csv')
        
        result = f"Best objective: {best_obj}\nIteration: {best_observation['iteration']}   Episode: {best_observation['episode']}" \
                + f"\nState: {list(map(int, best_observation['observation'].tolist()))}"
        print(result)
        write_to_file(result, self.results_path + self.name + '_result.txt')
        write_to_file(logs, self.results_path + self.name + '.csv')
        torch.save(self.model, 'models/' + self.name + '_last.pt')