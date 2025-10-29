import sys
sys.path.append('./') 
sys.path.insert(0, './policy/Diffusion-Policy') 

import fpsample
import torch  
import os
import re
from tasks import *
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union

import hydra
import dill
import yaml
import tyro
import gymnasium as gym
import numpy as np
import sapien
from omegaconf import OmegaConf 
from lightning.pytorch import LightningModule

from pathlib import Path
from collections import deque
import traceback
import importlib
from argparse import ArgumentParser

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import gym_utils
from utils.wrappers.record import RecordEpisodeMA
from utils.wrappers.suction_action import SuctionActionWrapper
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.dp_runner import DPRunner
from planner.motionplanner import PandaArmMotionPlanningSolver
from diffusion_policy.model.noposplat.encoder import get_encoder

# === NEW: pull the same helpers run.py is using ===
from configs.variant_manager import (
    apply_variant_to_yaml,
    get_wrapper_kwargs,
    get_robot_uids,
    get_tool_modes,
)

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """The environment ID of the task you want to simulate"""

    config: str = "configs/table/place_badminton.yaml"
    """Base configuration to build scenes, assets and agents (before variant patch)."""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "rgb"
    """Observation mode"""

    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    """Robot UID(s) to use. Usually inferred from --variant, so you can leave this None"""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'"""

    reward_mode: Optional[str] = None
    """Reward mode"""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode"""

    render_mode: str = "rgb_array"
    """Render mode"""

    shader: str = "default"
    """Shader pack for cameras. 'default', 'rt', 'rt-fast', etc."""

    record_dir: Optional[str] = './eval_video/DP2/{env_id}'
    """Directory to save recordings"""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If using human render mode, auto pauses the simulation upon loading"""

    quiet: bool = False
    """Disable verbose output."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 1000
    """Seed(s). Can be an int or a list. We'll sweep [seed, seed+100)"""

    data_num: int = 100
    """The number of episode data used for training the policy (just for logging)"""

    checkpoint_num: int = 300
    """The training epoch number of the checkpoint (just for logging)"""

    max_steps: int = 100
    """Maximum number of steps to run for each rollout"""

    ckpt: str = ''
    """Policy checkpoint"""

    exp_name: str = ''
    """Tag in log/video names"""

    variant: Annotated[str, tyro.conf.arg(aliases=["-v"])] = "gripper"
    """Choose baseline ('gripper') or ours ('ours')"""


def _normalize_task_name(raw: str) -> str:
    """
    Same logic as planner/run.py: convert CamelCase -> snake_case.
    e.g. 'PlaceBadminton' -> 'place_badminton'
    """
    if "_" in raw:
        return raw.lower()
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', raw)
    snake = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake


def get_policy(checkpoint, output_dir, device):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg = OmegaConf.create(cfg)
    model: LightningModule = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(payload['state_dict'])
    device = torch.device(device)
    policy = model.to(device)
    policy.eval()
    return policy


class DP:
    """
    Thin wrapper around (policy checkpoint -> DPRunner loop).
    DPRunner is basically the execution helper for diffusion-style visuomotor policies
    that output action sequences instead of single-step actions. :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, task_name, checkpoint_num: int, data_num: int, ckpt_path, id: int = 0):
        self.policy = get_policy(ckpt_path, None, 'cuda:0')
        self.runner = DPRunner(output_dir=None)

    def init_runner(self):
        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        return self.runner.get_action(self.policy, observation)

    def get_last_obs(self):
        return self.runner.obs[-1]
    
    def reset_policy(self):
        self.policy.reset()


def get_model_input(observation, agent_pos_list, agent_num):
    """
    Package observation into the dict format expected by DPRunner:
    - head_cam: CHW rgb from head camera(s)
    - agent_pos: concatenated joint + gripper states for all controlled agents
    """
    obs = {}
    # support single ('head_camera') and multi ('head_camera_global')
    if 'head_camera_global' in observation['sensor_data']:
        camera_name = 'head_camera_global'
    else:
        camera_name = 'head_camera'
    head_cam = np.moveaxis(
        observation['sensor_data'][camera_name]['rgb'].squeeze(0).cpu().numpy(),
        -1, 0
    ) / 255.0
    obs['head_cam'] = head_cam

    agent_pos = []
    for agent_id in range(agent_num):
        agent_pos.append(agent_pos_list[agent_id])
    obs['agent_pos'] = np.concatenate(agent_pos, axis=-1)
    return obs


def main(args: Args):
    np.set_printoptions(suppress=True, precision=5)

    # sanity on variant
    if args.variant not in ["gripper", "ours"]:
        raise ValueError(f"Unsupported variant={args.variant}, expected one of ['gripper','ours'].")

    verbose = 0
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])

    # parallel_in_single_scene flag (GUI optimization)
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in ["sensor_data", "rgb", "rgbd", "depth", "point_cloud"]:
        print("Disabling parallel single scene GUI render because obs_mode is visual.")
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False

    # ===== 1. Read base YAML and apply variant =====
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    raw_task_name = base_cfg["task_name"]
    task_name = _normalize_task_name(raw_task_name)

    # env_id fallback
    if args.env_id == "":
        env_id = base_cfg["task_name"] + "-rf"
    else:
        env_id = args.env_id

    # Patch the scene/robots/assets according to variant, same as run.py
    final_cfg = apply_variant_to_yaml(
        base_cfg,
        task_name.lower(),
        args.variant,
    )

    # figure out robot_uids / wrapper_kwargs / tool_modes from variant
    robot_uids_variant = get_robot_uids(task_name.lower(), args.variant)
    wrapper_kwargs = get_wrapper_kwargs(task_name.lower(), args.variant)
    tool_modes_raw = get_tool_modes(task_name.lower(), args.variant)

    # Planner expects either "gripper" (str) or ["gripper","stick"] (list)
    if isinstance(tool_modes_raw, (list, tuple)) and len(tool_modes_raw) == 1:
        tool_modes_for_solver = tool_modes_raw[0]
    else:
        tool_modes_for_solver = tool_modes_raw

    # dump patched cfg to a tmp yaml (like run.py does)
    os.makedirs('logs', exist_ok=True)
    tmp_cfg_path = os.path.join('logs', f"{task_name}_tmp.yaml")
    with open(tmp_cfg_path, "w") as f_tmp:
        yaml.safe_dump(final_cfg, f_tmp)

    # ===== logging setup =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"logs/dp2_{args.exp_name}_{env_id}_{args.data_num}_{args.checkpoint_num}_{timestamp}.txt"

    # init policy + runner
    dp = DP(env_id, args.checkpoint_num, args.data_num, args.ckpt)

    total_success = 0
    total_num = 0

    # We'll iterate seeds [seed0, seed0+100)
    for now_seed in range(args.seed[0], args.seed[0] + 100):

        # build env for this seed using the variant-adjusted config
        reward_mode = "dense" if args.reward_mode is None else args.reward_mode
        env = gym.make(
            env_id,
            config=tmp_cfg_path,
            obs_mode=args.obs_mode,
            reward_mode=reward_mode,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            robot_uids=robot_uids_variant if robot_uids_variant is not None else args.robot_uids,
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            viewer_camera_configs=dict(shader_pack=args.shader),
            num_envs=args.num_envs,
            sim_backend=args.sim_backend,
            enable_shadow=True,
            parallel_in_single_scene=parallel_in_single_scene,
        )

        # Wrap with the suction / stick / etc. logic determined by variant.
        # This is where we stop hardcoding suction_agents / stick_agents manually.
        env = SuctionActionWrapper(
            env,
            **wrapper_kwargs,
        )

        tool_suction = getattr(env.unwrapped, "_shared_suction_tool", None)
        assert tool_suction is not None, "SuctionTool not found; make sure SuctionActionWrapper is applied."
        tool_suction.clear_filters()
        tool_suction.allow_only_names(contains=["card"])          
        tool_suction.disallow_names(contains=["table", "floor", "ground", "cube", "Table", "Table-Workspace"])
        tool_suction = getattr(env.unwrapped, "_shared_circle_tool", None)

        tool_circle = getattr(env.unwrapped, "_shared_suction_tool", None)
        assert tool_circle is not None, "CircleTool not found; make sure SuctionActionWrapper is applied."
        tool_circle.clear_filters()
        tool_circle.allow_only_names(contains=["vase", "pokeball"])       
        tool_circle.disallow_names(contains=["table", "floor", "ground", "Table", "Table-Workspace"])

        # reset policy runner per new rollout
        dp.init_runner()
        dp.reset_policy()

        print("Current eval seed: ", now_seed)
        total_num += 1
        now_success = 0
        np.random.seed(now_seed)

        # optional video recorder
        record_dir = args.record_dir + f'_dp2_{args.exp_name}_' + str(timestamp) + '/' + str(now_seed)
        if record_dir:
            record_dir = record_dir.format(env_id=env_id)
            env = RecordEpisodeMA(
                env,
                record_dir,
                info_on_video=False,
                save_trajectory=False,
                max_steps_per_video=30000000
            )

        # ===== 2. Reset env, build planner =====
        raw_obs, _ = env.reset(seed=now_seed)

        # multi-agent or single-agent?
        planner_is_multi = isinstance(tool_modes_for_solver, (list, tuple))

        # base_pose(s) for planner
        try:
            base_pose = [agent.robot.pose for agent in env.agent.agents]
        except AttributeError:
            base_pose = env.unwrapped.agent.robot.pose

        planner = PandaArmMotionPlanningSolver(
            env,
            debug=False,
            vis=verbose,
            base_pose=base_pose,
            visualize_target_grasp_pose=verbose,
            print_env_info=False,
            is_multi_agent=planner_is_multi,
            tool_modes=tool_modes_for_solver
        )

        agent_num = planner.agent_num if planner_is_multi else 1

        # seeding env action space
        if now_seed is not None and env.action_space is not None:
            env.action_space.seed(now_seed)

        # first render / pause hook if we are in human mode
        if args.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = args.pause
            env.render()

        # ===== 3. Build initial obs for DP (agent_keys mapping etc.) =====
        spaces_is_dict = isinstance(env.action_space, gym.spaces.Dict)
        if spaces_is_dict:
            space_keys = list(env.action_space.spaces.keys())
        else:
            space_keys = []

        last_info = {}
        cnt = 0

        if planner_is_multi:
            # map agent_names in obs -> per-arm dof lengths
            obs_keys = list(raw_obs['agent'].keys())
            try:
                uid_list = [ag.uid for ag in env.agent.agents]
            except Exception:
                uid_list = obs_keys

            def _base(s):
                return s.split('-', 1)[0] if isinstance(s, str) else s

            agent_keys = []
            used = set()
            for idx in range(agent_num):
                uid = uid_list[idx] if idx < len(uid_list) else None
                k = uid if uid in obs_keys and uid not in used else None
                if k is None:
                    b = _base(uid) if uid is not None else None
                    k = next((x for x in obs_keys if _base(x) == b and x not in used), None)
                if k is None:
                    k = obs_keys[idx if idx < len(obs_keys) else -1]
                agent_keys.append(k)
                used.add(k)

            arm_dofs_map = {}
            for i, k in enumerate(agent_keys):
                ks = k if k in env.action_space.spaces else next(
                    (x for x in space_keys if _base(x) == _base(k)),
                    None
                )
                if ks is None:
                    ks = space_keys[i if i < len(space_keys) else 0]
                arm_dofs_map[k] = int(env.action_space.spaces[ks].shape[0] - 1)

            initial_qpos_list = []
            for i in range(agent_num):
                key = agent_keys[i]
                q = raw_obs['agent'][key]['qpos'].squeeze(0)
                q_arr = q.cpu().numpy() if hasattr(q, "cpu") else q.numpy()
                qpos_arm = q_arr[:arm_dofs_map[key]]
                initial_qpos = np.append(qpos_arm, planner.gripper_state[i])
                initial_qpos_list.append(initial_qpos)

            obs_model = get_model_input(raw_obs, initial_qpos_list, agent_num)
            dp.update_obs(obs_model)

        else:
            # single-arm case
            arm_dofs = int(env.action_space.shape[0] - 1)
            q = raw_obs['agent']['qpos'].squeeze(0)
            q_arr = q.cpu().numpy() if hasattr(q, "cpu") else q.numpy()
            qpos_arm = q_arr[:arm_dofs]
            initial_pos = np.append(qpos_arm, planner.gripper_state)
            obs_model = get_model_input(raw_obs, [initial_pos], agent_num=1)
            dp.update_obs(obs_model)

        # ===== 4. Closed-loop rollout =====
        while True:
            if verbose:
                print("Iteration:", cnt)
            cnt += 1
            if cnt > args.max_steps:
                break
            if cnt % 15 == 0:
                print("iter:", cnt)

            action = dp.get_action()

            if not planner_is_multi:
                # -------- single-agent --------
                if isinstance(action, dict):
                    if 'action_0' in action:
                        action_list = action['action_0']
                    elif 'action' in action:
                        action_list = action['action']
                    else:
                        action_list = list(action.values())[0]
                else:
                    action_list = action

                T = min(8, len(action_list))
                for i in range(T):
                    now_action = action_list[i]
                    raw_obs = env.get_obs()
                    if i == 0:
                        q = raw_obs['agent']['qpos'].squeeze(0)
                        q_arr = q.cpu().numpy() if hasattr(q, "cpu") else q.numpy()
                        cur_qpos = q_arr[:arm_dofs]
                    else:
                        cur_qpos = action_list[i - 1][:-1]

                    path = np.vstack((cur_qpos, now_action[:-1]))
                    try:
                        times, pos_traj, right_vel, acc, duration = planner.planner[0].TOPP(
                            path, 0.05, verbose=True
                        )
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        fallback = np.hstack([cur_qpos, now_action[-1]])
                        observation, reward, terminated, truncated, info = env.step(fallback)
                        last_info = info
                        continue

                    n_step = pos_traj.shape[0]
                    grip = now_action[-1]
                    if n_step == 0:
                        observation, reward, terminated, truncated, info = env.step(
                            np.hstack([cur_qpos, grip])
                        )
                        last_info = info
                    else:
                        for j in range(n_step):
                            true_action = np.hstack([pos_traj[j], grip])
                            observation, reward, terminated, truncated, info = env.step(true_action)
                            last_info = info

                        # after we finish executing that chunk, update obs for next DP query
                        obs_model = get_model_input(observation, [true_action], agent_num=1)
                        dp.update_obs(obs_model)

            else:
                # -------- multi-agent --------
                action_dict = defaultdict(list)
                action_step_dict = defaultdict(list)

                # break per-agent predicted horizon into interpolated joint trajectories
                for i_agent in range(agent_num):
                    key = agent_keys[i_agent]
                    # collect that agent's predicted horizon
                    action_list = []
                    for t in range(len(action[f'action_{i_agent}'])):
                        agent_action = action[f'action_{i_agent}'][t]
                        action_list.append(agent_action)

                    for i_h in range(8):
                        now_action = action_list[i_h]
                        raw_obs = env.get_obs()
                        if i_h == 0:
                            q = raw_obs['agent'][key]['qpos'].squeeze(0)
                            q_arr = q.cpu().numpy() if hasattr(q, "cpu") else q.numpy()
                            current_qpos = q_arr[:arm_dofs_map[key]]
                        else:
                            current_qpos = action_list[i_h - 1][:-1]

                        path = np.vstack((current_qpos, now_action[:-1]))
                        try:
                            times, position, right_vel, acc, duration = planner.planner[i_agent].TOPP(
                                path, 0.05, verbose=True
                            )
                        except Exception as e:
                            print(f"Error occurred: {e}")
                            action_now = np.hstack([current_qpos, now_action[-1]])
                            action_dict[key].append(action_now)
                            action_step_dict[key].append(1)
                            continue

                        n_step = position.shape[0]
                        action_step_dict[key].append(n_step)
                        gripper_state = now_action[-1]
                        if n_step == 0:
                            action_now = np.hstack([current_qpos, gripper_state])
                            action_dict[key].append(action_now)
                        for j in range(n_step):
                            true_action = np.hstack([position[j], gripper_state])
                            action_dict[key].append(true_action)
                
                # now play them in lockstep across agents
                start_idx = [0 for _ in range(agent_num)]
                for i_h in range(8):
                    max_step = 0
                    for i_agent in range(agent_num):
                        key = agent_keys[i_agent]
                        max_step = max(max_step, action_step_dict[key][i_h])
                    for j in range(max_step):
                        true_action = dict()
                        for i_agent in range(agent_num):
                            key = agent_keys[i_agent]
                            now_step = min(j, action_step_dict[key][i_h] - 1)
                            true_action[key] = action_dict[key][start_idx[i_agent] + now_step]
                        observation, reward, terminated, truncated, info = env.step(true_action)

                    if max_step == 0:
                        continue
                    action_concat = []
                    for i_agent in range(agent_num):
                        key = agent_keys[i_agent]
                        start_idx[i_agent] += action_step_dict[key][i_h]
                        action_concat.append(true_action[key])

                    if action_concat:
                        obs_model = get_model_input(observation, action_concat, agent_num)
                        dp.update_obs(obs_model)

            # ===== success check / render =====
            try:
                info_now = env.get_info()
            except Exception:
                info_now = last_info

            if args.render_mode is not None:
                env.render()

            success_flag = False
            if isinstance(info_now, dict):
                success_flag = bool(info_now.get('success', False))

            if success_flag:
                total_success += 1
                now_success = 1
                env.close()
                if record_dir:
                    print(f"Saving video to {record_dir}")
                print("success, step=", cnt)
                break

        # ===== per-seed summary logging =====
        success_rate_pct = 100.0 * total_success / total_num if total_num > 0 else 0.0
        with open(log_file, "a") as f_log:
            f_log.write(f"\n[Summary] Success Rate: {success_rate_pct:.2f}% ({total_success}/{total_num})\n")
            f_log.write(f"Current Seed: {now_seed}, success: {now_success}\n")

        if now_success == 0:
            print("failed")
            env.close() 
        if record_dir:
            print(f"Saving video to {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
