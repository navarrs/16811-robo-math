# ------------------------------------------------------------------------------
# @brief Tests the environment creation
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
from environment import Environment
from model import DQN
from common import ExperienceReplay, SETTINGS, Transition


#
# GLOBAL PARAMETERS
# ------------------------------------------------------------------------------
ENV_SETTINGS = {
    "grid_size": (5, 5),
    "world_size": (84, 84),
    "object_config": {
        "obstacles": 3, "goals": 1, "agents": 1
    }
}

CFG_SETTINGS = {
    "num_steps": 100,
}

env = Environment(ENV_SETTINGS)

grid = env.get_grid()
print(grid)

for i in range(CFG_SETTINGS["num_steps"]):
    action = np.random.randint(low=0, high=env.get_nactions())
    state, reward, done = env.step(action)
    print(f"Action: {env.get_action_name(action)} Reward: {reward} Done: {done}")
    
env.save()