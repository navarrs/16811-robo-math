# ------------------------------------------------------------------------------
# @brief Simple RL agent
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

#
# Methods
# ------------------------------------------------------------------------------


def select_action(state, eps, n_actions):
    f"""
    Chooses an action either randomly or from the policy.
    """
    global total_steps
    eps_thresh = eps["end"] + (eps["start"] - eps["end"]) * \
        math.exp(-1. * total_steps / eps["decay"])
    total_steps += 1
    if random.random() > eps_thresh:
        # print(f"From net")
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)  
    else:
        # print(f"Sample")
        action = torch.tensor([[random.randrange(n_actions)]], 
                              device=DEVICE, dtype=torch.long)
    return action


def optimize(settings):
    batch_size = settings["batch_size"]
    gamma = settings["gamma"]

    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)    
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE,
        dtype=torch.bool
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # print(state_batch.size(), action_batch.size(), reward_batch.size())

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size, device=DEVICE)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states.float()).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    # print(f"state-action values:\n{state_action_values}")
    # print(f"expected state-action values:\n{expected_state_action_values}")

    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def process_state(state, dim):
    # return np.reshape(state, dim)
    state = torch.from_numpy(state).permute(2, 1, 0)
    return state.unsqueeze(0)

#
# Main
# ------------------------------------------------------------------------------


for episode in range(n_episodes):
    print("Episode: {}/{}".format(episode, n_episodes))
    state = env.reset()
    state = process_state(state, dims)
    reward_all = 0.0
    for i in range(max_episode_len):
        action = select_action(state, eps, n_actions)
        action_name = SETTINGS["actions"][action]
        new_state, reward, done = env.step(action)
        reward_all += reward
        reward = torch.tensor([reward], device=DEVICE, dtype=torch.float32)
        
        if done:
            next_state = None
        else:
            new_state = process_state(new_state, dims)

        memory.push(state, action, new_state, reward)

        optimize(SETTINGS)

        if done:
            print(f"Episode Done")
            break
        
        if i % SETTINGS["log_interval"] == 0:
            print("Step: {}/{} Action: {} Reward: {}"
                .format(i+1, max_episode_len, action_name, reward_all))
    
    if episode % SETTINGS["target_update"] == 0:
        target_net.load_state_dict(policy_net.state_dict())

print(f"Training Complete!")


#         plt.imshow(new_state, interpolation="nearest")
#         plt.savefig(f"out/state_{i+1}.png")


#         print("Step: {}/{} Action: {}".format(i, MAX_EPISODE_LENGTH, action_name))
