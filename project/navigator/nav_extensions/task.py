import attr
import gzip
import json
import os

from typing import List, Optional, Dict, Type, Any
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.registry import registry
from habitat.core.utils import (
    DatasetFloatJSONEncoder,
    not_none_validator
)
from habitat.datasets.pointnav.pointnav_dataset import (
    PointNavDatasetV1
)
from habitat.tasks.nav.nav import (
    NavigationGoal,
    NavigationEpisode,
    NavigationTask,
    merge_sim_episode_config
)

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"

def merge_sim_episode_with_heading_config(
    sim_config: Config, episode: Type[Episode]) -> Any:
    sim_config = merge_sim_episode_config(sim_config, episode)
    # sim_config.defrost()
    # sim_cfg.objects = [episode.objects.__dict__]
    return sim_config

@registry.register_task(name="NavigatorTask-v0")
class NavigatorTask(NavigationTask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_heading_config(sim_config, episode)


@attr.s(auto_attribs=True, kw_only=True)
class NavigatorGoal:
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    radius: float = attr.ib(default=None, validator=not_none_validator)
    geodesic_distance: float = attr.ib(
        default=None, validator=not_none_validator)
    info: Optional[Dict[str, str]] = attr.ib(default=None)

#
# Navigator dataset
# ------------------------------------------------------------------------------


@attr.s(auto_attribs=True, kw_only=True)
class NavigatorEpisode(NavigationEpisode):
    goals: NavigatorGoal = attr.ib(default=None, validator=not_none_validator)


@registry.register_dataset(name="NavigatorDataset-v0")
class NavigatorDatasetv0(PointNavDatasetV1):
    episodes: List[NavigatorEpisode]

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)

        for i, episode in enumerate(deserialized):
            nav_episode = NavigatorEpisode(**episode)

            if scenes_dir is not None:
                if nav_episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    nav_episode.scene_id = (
                        nav_episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX):
                        ]
                    )

                nav_episode.scene_id = os.path.join(
                    scenes_dir, nav_episode.scene_id
                )

            if nav_episode.goals is not None:
                for g, goal in enumerate(nav_episode.goals):
                    # Change this to navigation goal
                    nav_episode.goals[g] = NavigatorGoal(**goal)

            self.episodes.append(nav_episode)
