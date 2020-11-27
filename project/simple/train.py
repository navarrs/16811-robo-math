
from __future__ import division

import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.misc
import os
from collections import namedtuple

from grid_world import gameEnv
env = gameEnv(partial=False, size=5)

ACTIONS = ["up", "down", "right", "left"]
WORLD_DIMS = [84, 84, 3]
FLATTENED_DIMS =  WORLD_DIMS[0] * WORLD_DIMS[1] * WORLD_DIMS[2]
NUM_EPISODES = 1
MAX_EPISODE_LENGTH = 10
PRETRAIN_STEPS = 5
total_steps = 0
annealing_steps = 2
e, starte, ende = 1, 1, 0.1
drop_e = (starte-ende) / annealing_steps
update_freq = 5

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ExperienceReplay(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position= 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class DQN(nn.Module):
    def __init__(self, height, width, channels, actions):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        
        def conv2d_size_out(size, ksize=5, stride=2):
            return (size - (ksize-1) -1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, actions)
        
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        return self.head(x.view(x.size(0), -1))
        
def process_state(state):
    return np.reshape(state, FLATTENED_DIMS)

policy_net = DQN(WORLD_DIMS[0], WORLD_DIMS[1], WORLD_DIMS[2], len(ACTIONS))
target_net = DQN(WORLD_DIMS[0], WORLD_DIMS[1], WORLD_DIMS[2], len(ACTIONS))
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

for episode in range(NUM_EPISODES):
    print("Episode: {}/{}".format(episode, NUM_EPISODES))
    state = torch.from_numpy(env.reset()).permute(2, 1, 0)
    # # state = process_state(state)
    state = state.unsqueeze(0)
    done = False
    reward_all = 0
    for i in range(MAX_EPISODE_LENGTH):
       
        # Compute action
        if np.random.rand(1) < e or total_steps < PRETRAIN_STEPS:
            action_num = np.random.randint(low=0, high=len(ACTIONS))
        else:
            action_prob = net(state)
            _, action_num = torch.max(action_prob, 1)
        action_name = ACTIONS[action_num]
        
        # Compute step
        new_state, reward, done = env.step(action_num)
        total_steps += 1
        
        if total_steps > PRETRAIN_STEPS:
            if e > ende:
                e -= drop_e
            
            if total_steps % update_freq == 0:
                pass
        
        plt.imshow(new_state, interpolation="nearest")
        plt.savefig(f"out/state_{i+1}.png")
        
        
        print("Step: {}/{} Action: {}".format(i, MAX_EPISODE_LENGTH, action_name))
    
        
