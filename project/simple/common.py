# ------------------------------------------------------------------------------
# @brief Simple RL agent
# @author navarrs
# ------------------------------------------------------------------------------

#
# INCLUDES
# ------------------------------------------------------------------------------
import random
from collections import namedtuple

#
# GLOBAL PARAMETERS
# ------------------------------------------------------------------------------
SETTINGS = {
    "actions": ["up", "down", "left", "right"],
    "world_size": 5,
    "world_dims": [84, 84, 3],
    "num_episodes": 10000,
    "log_interval": 5,
    "max_episode_length": 50,
    "pre_train_steps": 10000,
    "eps": {"start": 0.9, "end": 0.05, "decay": 10000, "now": 0.9},
    "gamma": 0.999,
    "target_update": 4,
    "batch_size": 32,
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

#
# METHODS
# ------------------------------------------------------------------------------

#
# CLASSES
# ------------------------------------------------------------------------------


class ExperienceReplay(object):
    f"""
        Buffer that holds recent transitions and also samples random transitions.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
