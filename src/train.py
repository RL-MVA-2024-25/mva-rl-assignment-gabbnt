from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
from torch import nn
import numpy as np
import random
from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
env_random = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)
# Parameters
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.99,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 10000,
          'epsilon_delay_decay': 500,
          'batch_size': 512,
          'gradient_steps': 2,
          'update_target_strategy': 'ema', # or 'replace'
          'update_target_freq': 200,
          'update_target_tau': 0.001,
          'use_Huber_loss': False,
          'criterion': torch.nn.SmoothL1Loss()}

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
class ProjectAgent():
    def __init__(self, to_be_trained=False):
        self.envs=[env,env_random]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'],self.device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.total_steps = 0
        self.max_episode=300
        self.nb_neurons=500
        self.state_dim = self.envs[0].observation_space.shape[0]
        self.n_action = self.envs[0].action_space.n 
        self.model = torch.nn.Sequential(nn.Linear(self.state_dim, self.nb_neurons),
                                nn.ReLU(),
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(self.nb_neurons, self.n_action)).to(self.device)
         
        if config['use_Huber_loss']:
            self.criterion = torch.nn.SmoothL1Loss()
        else:
            self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_gradient_steps = config['gradient_steps']
        self.target_model = deepcopy(self.model).to(self.device)
        self.update_target_strategy = config['update_target_strategy']
        self.update_target_freq = config['update_target_freq']
        self.update_target_tau = config['update_target_tau']
        if to_be_trained: # Must be switched to True for training
            self.train()
        else:
            self.load()
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def act(self, observation, use_random=False):
        if use_random:
            return random.randrange(self.n_action)
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()
    
    def train(self):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = self.envs[0].reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < self.max_episode:
            # update epsilon
            if step+200*episode > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = np.random.randint(self.nb_actions)
            else:
                action = self.act(state)
            # step
            next_state, reward, done, trunc, _ = self.envs[episode % 2].step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                print(f"Episode {episode}")
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = self.envs[episode%2].reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
        self.save()
        return episode_return

    def save(self, path='DQN_weights.pth'):
        torch.save(self.model.state_dict(), path)

    def load(self, path='WDQN_weights.pth'):
        self.model.load_state_dict(torch.load(path,map_location=torch.device(self.device)))
