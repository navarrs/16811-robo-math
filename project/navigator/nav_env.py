import os
import random
from typing import Optional, Type

import habitat
from habitat import Config, Dataset
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv


#
# Classes
# ------------------------------------------------------------------------------


@baseline_registry.register_env(name="NavigatorRLEnv")
class NavigatorRLEnv(NavRLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._previous_action = None
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        self._prev_measure = {
            "distance_to_goal": 0.0,
        }
        super().__init__(config, dataset)

    def get_reward_range(self):
        return (self._rl_config.SLACK_REWARD - 1.0, 
                self._rl_config.SUCCESS_REWARD + 1.0)

    def reset(self):
        self._previous_action = None
        observations = super().reset()
        self._prev_measure["distance_to_goal"] = self._env.get_metrics()[
            "distance_to_goal"]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD
        agent_to_goal_reward = self.get_agent_to_goal_reward()
        episode_success_reward = 0
        
        collision_reward = 0
        if self._rl_config.COLLISION_CHECK:
            collision_reward = self.get_collision_reward()

        action_name = self._env.task.get_action_name(
            self._previous_action["action"]
        )

        if self._episode_success(observations):
            episode_success_reward = self._rl_config.SUCCESS_REWARD

        reward += (
            agent_to_goal_reward +
            collision_reward +
            episode_success_reward
        )
        return reward

    def get_agent_to_goal_reward(self):
        curr_metric = self._env.get_metrics()["distance_to_goal"]
        prev_metric = self._prev_measure.get("distance_to_goal")
        dist_reward = prev_metric - curr_metric
        self._prev_measure["distance_to_goal"] = curr_metric
        return dist_reward
    
    def get_collision_reward(self):
        collision_dist = self._env.get_metrics()["collision_distance"]
        if collision_dist <= self._rl_config.COLLISION_THRESH:
            return self._rl_config.COLLISION_REWARD
        return 0.0

    def _episode_success(self, observations):
        dist = self._env.get_metrics()["distance_to_goal"]
        return abs(dist) < self._success_distance

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success(observations):
            done = True
        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        info["episode_success"] = self._episode_success(observations)
        return info