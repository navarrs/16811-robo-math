# ------------------------------------------------------------------------------
# @brief Tests the environment
# @author navarrs
# ------------------------------------------------------------------------------

#
# INCLUDES
# ------------------------------------------------------------------------------
from __future__ import division

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.misc
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#
# MY INCLUDES
# ------------------------------------------------------------------------------
from grid_world import gameEnv
from model import DQN
from common import ExperienceReplay, SETTINGS, Transition


#
# GLOBAL PARAMETERS
# ------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print(f"Using device: {DEVICE}")
print(f"Settings:\n{SETTINGS}")


n_actions = len(SETTINGS["actions"])
n_episodes = SETTINGS["num_episodes"]
max_episode_len = SETTINGS["max_episode_length"]
dims = SETTINGS["world_dims"]
eps = SETTINGS["eps"]

policy_net = DQN(dims[0], dims[1], dims[2], n_actions).to(DEVICE)
target_net = DQN(dims[0], dims[1], dims[2], n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ExperienceReplay(100)

total_steps = 0

env = gameEnv(partial=False, size=SETTINGS["world_size"])