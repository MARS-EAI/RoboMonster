from typing import Any, Dict, Tuple

import numpy as np
import sapien
import torch
import json
import yaml
import random

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.utils.structs.pose import Pose
import utils.scenes

@register_env("PlaceBadminton-rf", max_episode_steps=500)
class PlaceBadmintonEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("panda", "panda"), ("panda", "panda_stick")]
    # agent: MultiAgent[Tuple[Panda, Panda]]

    goal_thresh = 0.025
    cube_color = np.concatenate((np.array([187, 116, 175]) / 255, [1]))
    light_cube_color = np.concatenate((np.array([187, 116, 175]) / 255, [0.5]))
    cube_half_size = 0.02

    def __init__(
        self, *args, robot_uids=("panda", "panda"), robot_init_qpos_noise=0.02, **kwargs # robot_uid has been init in gym.make() 
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        assert 'config' in kwargs
        with open(kwargs['config'], 'r', encoding='utf-8') as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        del kwargs['config']
        self.r2id = robot_uids[1]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
        sensor_cfg = camera_cfg.get('sensor', {})
        all_camera_configs =[]
        if sensor_cfg:
            for sensor in sensor_cfg:
                pose = sensor['pose']
                if pose['type'] == 'pose':
                    sensor['pose'] = sapien.Pose(*pose['params'])
                elif pose['type'] == 'look_at':
                    sensor['pose'] = sapien_utils.look_at(*pose['params'])
                all_camera_configs.append(CameraConfig(**sensor))
        return all_camera_configs


    @property
    def _default_human_render_camera_configs(self):
        camera_cfg = self.cfg.get('cameras', {})
        render_cfg = camera_cfg.get('human_render', {})
        all_camera_configs =[]
        if render_cfg:
            for render in render_cfg:
                pose = render['pose']
                if pose['type'] == 'pose':
                    render['pose'] = sapien.Pose(*pose['params'])
                elif pose['type'] == 'look_at':
                    render['pose'] = sapien_utils.look_at(*pose['params'])
                all_camera_configs.append(CameraConfig(**render))
        return all_camera_configs

    
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=20,
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )
    
    def _load_agent(self, options: dict):
        init_poses = []
        for agent_cfg in self.cfg['agents']:
            init_poses.append(sapien.Pose(p=agent_cfg['pos']['ppos']['p']))
        super()._load_agent(options, init_poses)

    def _load_scene(self, options: dict):
        scene_name = self.cfg['scene']['name']
        scene_builder = getattr(utils.scenes, f'{scene_name}SceneBuilder')
        self.scene_builder = scene_builder(env=self, cfg=self.cfg)
        self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            i = int(env_idx[0].item())
            # self.badminton_num = int(self._batched_episode_rng[i].randint(6, 8))
            # self.badminton_num = 6
            self.badminton_num = int(self._batched_episode_rng[i].choice([6, 7], p=[0.7, 0.3]))
            print(self.badminton_num)
            # print(self.r2id)


    def evaluate(self):
        badminton_num = self.badminton_num
        error = self.badminton.pose.p - self.barrel.pose.p
        # print(self.badminton.pose.p)
        # print(self.barrel.pose.p)
        # print(badminton_num)
        if badminton_num == 6:
            success = (error[..., 0] <= 0.34) and (self.barrel.pose.p[..., 0] < -0.355)
        elif badminton_num == 7:
            success = (error[..., 0] <= 0.45) and (self.barrel.pose.p[..., 0] < -0.355)
        return {
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        return {}

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return {}
