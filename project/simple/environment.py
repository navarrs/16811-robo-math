# ------------------------------------------------------------------------------
# @brief RL environment
# @author navarrs
# ------------------------------------------------------------------------------

#
# INCLUDES
# ------------------------------------------------------------------------------
import numpy as np
import copy
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from enum import Enum

#
# Classes
# ------------------------------------------------------------------------------


class ObjectType(Enum):
    OBSTACLE = 1
    GOAL = 2
    AGENT = 3


class Coordinate(object):
    X = 0.0
    Y = 0.0


class Object():
    def __init__(self, obj_type, position, reward=None):
        f"""
        Initializes an object that will be used in the environment.
        Args
        ----
            obj_type: type of object in the environment [OBSTACLE, GOAL, AGENT]
            position: position where the object is in the environent
            reward:   reward assigned to the object
        """
        self.obj_type = obj_type
        self.position = position
        self.reward = reward

    #
    # MEMBER SETTERS
    # --------------------------------------------------------------------------
    def set_position(self, position):
        self.position = position
    
    #
    # MEMBER GETTERS
    # --------------------------------------------------------------------------
    def get_position(self):
        return self.position
    
    def get_reward(self):
        return self.reward

    def get_obj_type(self):
        return self.obj_type


class Environment():
    def __init__(self, settings):
        f"""
        Initializes the environment.
        Args
        ----
            settings:   Configuration file containing the settings to create the
                        environment:
                            - grid_size:  number of cells that the world contains
                            - world_size: dimensions used to resize the grid
                            - object_config: number of goals, obstacles and agents 
        """
        self.world_size = settings["world_size"]
        self.grid_size = settings["grid_size"]
        self.grid = np.zeros(self.grid_size, dtype=np.int)
        self.actions = ["up", "down", "left", "right"]
        self.object_config = settings["object_config"]
        self.n_actions = len(self.actions)
        self.objects = []
        self.agent = None
        self.states = []
        self.reset()

    def reset(self):
        f"""
        Create and spawn objects in the world.
        Out
        ---
            state: initial state of the episode.
        """
        self.states = []
        for o in range(self.object_config["obstacles"]):
            obj = self.spawn(obj_type=ObjectType.OBSTACLE, reward=-1)
            self.objects.append(obj)

        for g in range(self.object_config["goals"]):
            obj = self.spawn(obj_type=ObjectType.GOAL, reward=1)
            self.objects.append(obj)

        self.agent = self.spawn(ObjectType.AGENT)

        state = self.render()
        self.states.append(state)
        plt.imshow(state, interpolation="nearest")
        plt.savefig(f"out/reset_state.png")
        return state

    def spawn(self, obj_type, reward=None):
        f"""
        Spawn an object at a random and free location on the grid.
        Args
        ----
            obj_type:   type of object (ObjectType) to spawn
            reward:     reward value assigned to this object
        Out
        ---
            obj: the spawned object
        """
        free = np.argwhere(self.grid == 0)
        obj_location = free[np.random.choice(range(len(free)), size=1)][0]
        self.grid[obj_location[0], obj_location[1]] = obj_type.value
        obj = Object(obj_type, obj_location, size, reward)
        # print(obj.get_position(), obj.get_reward())
        return obj

    def render(self):
        f"""
        Renders a state.
        Out
        ---
            state: new state of the episode.
        """
        state = np.zeros(
            (self.grid_size[0], self.grid_size[1], 3), dtype=np.float32)

        for obj in self.objects:
            pos = obj.get_position()
            val = obj.get_obj_type().value-1
            state[pos[0], pos[1], val] = 1

        pos = self.agent.get_position()
        val = self.agent.get_obj_type().value-1
        state[pos[0], pos[1], val] = 1

        state = resize(
            state, (self.world_size[0], self.world_size[1]), order=0)
        return state

    def step(self, action):
        f"""
        Performs a step for the agent and a state update. 
        Args
        ----
            action: action that the agent takes: [up, down, left, right]
        Out
        ---
            state: new state of the episode.
            reward: obtained reward (or penalty) from this step
            done: True if the agent reached the goal
        """
        reward, done = self.move(action)
        state = self.render()
        self.states.append(state)
        return state, reward, done

    def move(self, action):
        f"""
        Moves the agent based on the input action.
        Args
        ----
            action: action that the agent takes: [up, down, left, right]
        Out
        ---
            reward: obtained reward (or penalty) from this movement
            done: True if the agent reached the goal
        """
        reward = 0
        done = False
        action_val = self.actions[action]

        agent_position = self.agent.get_position()
        new_agent_position = agent_position.copy()

        if action_val == "up":
            if agent_position[0] == 0:
                return -0.001, False
            new_agent_position[0] -= 1

        elif action_val == "down":
            if agent_position[0] == self.grid_size[0]-1:
                return -0.001, False
            new_agent_position[0] += 1

        elif action_val == "left":
            if agent_position[1] == 0:
                return -0.001, False
            new_agent_position[1] -= 1

        elif action_val == "right":
            if agent_position[1] == self.grid_size[1]-1:
                return -0.001, False
            new_agent_position[1] += 1

        # print(f"\nCurrent agent's location: {agent_position} new {new_agent_position}")
        for obj in self.objects:
            if (
                obj.get_position()[0] == new_agent_position[0] and
                obj.get_position()[1] == new_agent_position[1]
            ):
                reward += obj.get_reward()

                if ObjectType.GOAL == obj.get_obj_type():
                    done = True

        self.agent.set_position(new_agent_position)
        return reward, done

    def save(self, outfile="out/game.avi", FPS=12):
        f"""
        Renders states into a video.
        Args
        ----
            FPS: frames per second used to write the video
            outfile: filename to save this video.
        """
        import cv2
        out = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'DIVX'),
                              FPS, self.world_size)
        for s in self.states:
            s = (255*s).astype(np.uint8)
            s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
            out.write(s)
        out.release()

    #
    # MEMBER GETTERS
    # --------------------------------------------------------------------------
    def get_grid(self):
        return self.grid

    def get_nactions(self):
        return self.n_actions

    def get_action_name(self, action):
        return self.actions[action]

    def get_states(self):
        return self.states
