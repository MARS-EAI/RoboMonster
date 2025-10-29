import sys
sys.path.append('./')
sys.path.insert(0, './policy/Diffusion-Policy')

import os
import re
import time
import dill
import yaml
import torch
import tyro
import hydra
import numpy as np
import gymnasium as gym
import sapien
from dataclasses import dataclass
from typing import List, Optional, Annotated, Union
from datetime import datetime
from omegaconf import OmegaConf
from collections import defaultdict
from tasks import *
from mani_skill.envs.sapien_env import BaseEnv
from utils.wrappers.suction_action import SuctionActionWrapper
from utils.wrappers.record import RecordEpisodeMA
from diffusion_policy.env_runner.dp_runner import DPRunner
from planner.motionplanner import PandaArmMotionPlanningSolver
import fpsample

# === variant helpers (same idea as planner/run.py) ===
from configs.variant_manager import (
    apply_variant_to_yaml,
    get_wrapper_kwargs,
    get_robot_uids,
    get_tool_modes,
)

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = ""
    """Environment ID to eval on. If empty, we'll infer <task_name>-rf from the yaml."""

    config: str = "configs/table/place_badminton.yaml"
    """Base YAML before variant patch."""

    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "pointcloud"
    """DP3 usually consumes 3D point cloud obs (sparse sampled)."""

    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    """Simulation backend: 'auto' / 'cpu' / 'gpu'."""

    reward_mode: Optional[str] = None
    """Reward mode. If None we default to 'dense' when building the env."""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Vectorized envs. We keep 1 for rollout."""

    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    """Control mode (dp_joint_pos / pd_ee_pose etc.)."""

    render_mode: str = "rgb_array"
    """'rgb_array', 'sensors', or 'human'."""

    shader: str = "default"
    """Shader pack for cameras. 'default', 'rt', 'rt-fast', ..."""

    record_dir: Optional[str] = './eval_video/DP3/{env_id}'
    """Directory to save rollout videos."""

    pause: Annotated[bool, tyro.conf.arg(aliases=["-p"])] = False
    """If GUI viewer is open (render_mode='human'), start paused or not."""

    quiet: bool = False
    """Silence debug prints (not heavily used yet)."""

    seed: Annotated[Optional[Union[int, List[int]]], tyro.conf.arg(aliases=["-s"])] = 1000
    """Eval starts from this seed and increments; we roll multiple seeds."""

    max_steps: int = 60
    """Safety cap per episode (outer while loop)."""

    ckpt: str = ''
    """DP3 checkpoint path (policy weights)."""

    exp_name: str = ''
    """Optional experiment tag used in log/video naming."""

    variant: Annotated[str, tyro.conf.arg(aliases=["-v"])] = "ours"
    """'gripper' baseline (single-end-effector) vs 'ours' (heterogeneous multi-end-effectors)."""

def _normalize_task_name(raw: str) -> str:
    """
    Turn "PlaceBadminton" -> "place_badminton"
    Turn "CircleVase" -> "circle_vase"
    If it's already snake_case, just lower it.
    """
    if "_" in raw:
        return raw.lower()
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', raw)
    snake = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake


def get_policy(checkpoint: str, device: str):
    """
    Load DP3 checkpoint. DP3 = 3D Diffusion Policy.
    DP3 encodes point clouds into a compact 3D latent, then
    denoises noise into a *sequence of future actions*, not just one step. :contentReference[oaicite:2]{index=2}
    """
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = OmegaConf.create(payload['cfg'])
    # hydra.instantiate(cfg.policy) builds the LightningModule-ish policy
    model = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(payload['state_dict'])
    policy = model.to(torch.device(device))
    policy.eval()
    return policy


class DPWrapper:
    """
    Light wrapper so we can re-init runner per episode.
    DPRunner handles the "receding horizon" inference loop:
    - keep internal obs history
    - ask policy for N-step action chunks
    This chunked-action rollout style is typical for diffusion-style visuomotor policies:
    you generate a short horizon of actions, execute them with smoothing, then replan. :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, ckpt_path: str):
        self.policy = get_policy(ckpt_path, 'cuda:0')
        self.runner = DPRunner(output_dir=None)

    def init_runner(self):
        self.runner = DPRunner(output_dir=None)

    def update_obs(self, observation):
        self.runner.update_obs(observation)

    def get_action(self, observation=None):
        return self.runner.get_action(self.policy, observation)

    def reset_policy(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()


def get_model_input(observation, agent_pos_list, agent_num):
    """
    Build the observation dict fed into DPRunner:
    - pointcloud: downsampled xyz (N,3)
    - agent_pos: concat of each agent's [joint_positions..., gripper_state]
    This matches how your multi-dp3 / single-dp3 scripts sampled + fused.
    DP3 specifically conditions on (point cloud embedding + robot joint pose). :contentReference[oaicite:4]{index=4}
    """
    obs = {}

    # point cloud comes as observation["pointcloud"]["xyzw"]: (1, M, 4)
    pc_full = observation["pointcloud"]["xyzw"].squeeze(0).cpu().numpy()
    # mask valid points: channel -1 is valid flag; also z>2.5mm to drop table floor noise
    mask = (pc_full[:, -1] == 1) & (pc_full[:, 2] > 0.0025)
    # also clip to first 256*256 points to avoid weird padded tail
    mask[:-256*256] = False
    xyz = pc_full[mask, :3]

    # fps / kdline sampling -> ~1024 points
    kd_idx = fpsample.bucket_fps_kdline_sampling(xyz, 1024, h=9)
    point_cloud = xyz[kd_idx]
    obs['pointcloud'] = point_cloud

    # robot joint + gripper states
    agent_pose_list = []
    for agent_id in range(agent_num):
        agent_pose_list.append(agent_pos_list[agent_id])
    obs['agent_pos'] = np.concatenate(agent_pose_list, axis=-1)

    return obs

def main(args: Args):
    np.set_printoptions(suppress=True, precision=5)
    verbose = 0

    # normalize seed form
    if isinstance(args.seed, int):
        args.seed = [args.seed]
    if args.seed is not None:
        np.random.seed(args.seed[0])

    # ---------------- variant-inject the YAML ----------------
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    raw_task_name = base_cfg["task_name"]
    task_name = _normalize_task_name(raw_task_name)

    # if env_id wasn't given, follow run.py logic: "<TaskName>-rf"
    if args.env_id == "":
        env_id = base_cfg["task_name"] + "-rf"
    else:
        env_id = args.env_id

    # patch the yaml based on variant
    # - e.g. 'gripper' => single-arm panda gripper only
    # - 'ours' => heterogeneous multi-end-effectors, e.g. gripper + stick/suction arm
    # this mirrors planner/run.py, so we get consistent robot setups.
    final_cfg = apply_variant_to_yaml(
        base_cfg,
        task_name.lower(),
        args.variant,
    )

    # figure out robot_uids / wrapper kwargs / tool modes
    robot_uids_variant = get_robot_uids(task_name.lower(), args.variant)
    wrapper_kwargs = get_wrapper_kwargs(task_name.lower(), args.variant)
    tool_modes_raw = get_tool_modes(task_name.lower(), args.variant)

    # planner wants either a single string ('gripper') or a list ['gripper','stick']
    if isinstance(tool_modes_raw, (list, tuple)) and len(tool_modes_raw) == 1:
        tool_modes_for_solver = tool_modes_raw[0]
    else:
        tool_modes_for_solver = tool_modes_raw

    # write final_cfg to a tmp yaml so gym.make can ingest it
    os.makedirs('logs', exist_ok=True)
    tmp_cfg_path = os.path.join('logs', f"{task_name}_tmp.yaml")
    with open(tmp_cfg_path, "w") as f_tmp:
        yaml.safe_dump(final_cfg, f_tmp)

    # ---------------- viewer / parallel flags ----------------
    parallel_in_single_scene = args.render_mode == "human"
    if args.render_mode == "human" and args.obs_mode in [
        "sensor_data", "rgb", "rgbd", "depth", "point_cloud", "pointcloud"
    ]:
        # ManiSkill quirk: GUI can't easily batch-render full visual obs in one scene.
        parallel_in_single_scene = False
    if args.render_mode == "human" and args.num_envs == 1:
        parallel_in_single_scene = False

    # reward mode default
    reward_mode = "dense" if args.reward_mode is None else args.reward_mode

    # ---------------- logging paths ----------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = f"logs/dp3_{args.exp_name}_{env_id}_{timestamp}.txt"

    # policy wrapper
    dp = DPWrapper(args.ckpt)

    total_success = 0
    total_num = 0

    # sweep seeds [seed0, seed0+100)
    for now_seed in range(args.seed[0], args.seed[0] + 100):

        # reset DPRunner & policy hidden state for this rollout
        dp.init_runner()
        dp.reset_policy()

        # ------------- build env for this seed -------------
        env = gym.make(
            env_id,
            config=tmp_cfg_path,
            obs_mode=args.obs_mode,
            reward_mode=reward_mode,
            control_mode=args.control_mode,
            render_mode=args.render_mode,
            robot_uids=robot_uids_variant,
            sensor_configs=dict(shader_pack=args.shader),
            human_render_camera_configs=dict(shader_pack=args.shader),
            viewer_camera_configs=dict(shader_pack=args.shader),
            num_envs=args.num_envs,
            sim_backend=args.sim_backend,
            enable_shadow=True,
            parallel_in_single_scene=parallel_in_single_scene,
        )

        # Wrap action space into heterogeneous tool logic.
        # wrapper_kwargs (from variant) decides things like:
        #   on_threshold, stick_agents=['panda_stick-1'], gripper_hold_value=-1.0, etc.
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

        # -- optional suction filter tuning per task --
        # tool = getattr(env.unwrapped, "_shared_suction_tool", None)
        # if tool is not None:
        #     tool.clear_filters()
        #     tool.allow_only_names(contains=["card"])
        #     tool.disallow_names(contains=["table", "floor", "ground", "cube"])

        print("Current eval seed:", now_seed)
        total_num += 1
        now_success = 0
        np.random.seed(now_seed)

        # attach recorder (video dump)
        record_dir = None
        if args.record_dir:
            record_dir = args.record_dir + f'_dp3_{args.exp_name}_' + str(timestamp) + '/' + str(now_seed)
            record_dir = record_dir.format(env_id=env_id)
            env = RecordEpisodeMA(
                env,
                record_dir,
                info_on_video=False,
                save_trajectory=False,
                max_steps_per_video=30_000_000
            )

        # reset env
        raw_obs, _ = env.reset(seed=now_seed)

        # is this rollout multi-agent or single-agent?
        # heuristic: if tool_modes is a list/tuple with >1 entries, it's multi;
        # otherwise it's single.
        planner_is_multi = isinstance(tool_modes_for_solver, (list, tuple))

        # base pose(s) for PandaArmMotionPlanningSolver
        # ManiSkill often has env.agent.agents list when multi-arm. :contentReference[oaicite:5]{index=5}
        try:
            base_pose = [agent.robot.pose for agent in env.agent.agents]
        except AttributeError:
            base_pose = env.unwrapped.agent.robot.pose

        # motion planner wrapper: does joint-space TOPP for each arm/tool
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

        # seed env.action_space for determinism
        if now_seed is not None and env.action_space is not None:
            env.action_space.seed(now_seed)

        # initial manual render / pause hook (if GUI)
        if args.render_mode is not None:
            viewer = env.render()
            if isinstance(viewer, sapien.utils.Viewer):
                viewer.paused = args.pause
            env.render()

        last_info = {}
        cnt = 0

        # We have two branches here mostly to compute:
        #   - agent_keys mapping for multi-agent
        #   - arm DOF per agent
        spaces_is_dict = isinstance(env.action_space, gym.spaces.Dict)
        if spaces_is_dict:
            space_keys = list(env.action_space.spaces.keys())
        else:
            space_keys = []

        if planner_is_multi:
            # map from env observations to each controlled arm
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

            # build the "agent_pos" vector(s) fed to DP3
            initial_qpos_list = []
            for aid in range(agent_num):
                key = agent_keys[aid]
                q = raw_obs['agent'][key]['qpos'].squeeze(0)
                q_arr = q.cpu().numpy() if hasattr(q, "cpu") else q.numpy()
                qpos_arm = q_arr[:arm_dofs_map[key]]
                initial_qpos = np.append(qpos_arm, planner.gripper_state[aid])
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

        while True:
            if verbose:
                print("Iteration:", cnt)
            cnt += 1
            if cnt > args.max_steps:
                break
            if cnt % 15 == 0:
                print("iter:", cnt)

            # ask DP3 for an action *sequence* (horizon H)
            action = dp.get_action()

            if not planner_is_multi:
                # ================= SINGLE-AGENT EXECUTION =================
                if isinstance(action, dict):
                    if 'action_0' in action:
                        action_list = action['action_0']
                    elif 'action' in action:
                        action_list = action['action']
                    else:
                        action_list = list(action.values())[0]
                else:
                    action_list = action

                # we only take first up to 8 steps of that predicted horizon
                T = min(8, len(action_list))
                arm_dofs = int(env.action_space.shape[0] - 1)

                for i_step in range(T):
                    now_action = action_list[i_step]
                    raw_obs = env.get_obs()

                    # current joint pos for TOPP start
                    if i_step == 0:
                        q = raw_obs['agent']['qpos'].squeeze(0)
                        q_arr = q.cpu().numpy() if hasattr(q, "cpu") else q.numpy()
                        cur_qpos = q_arr[:arm_dofs]
                    else:
                        cur_qpos = action_list[i_step - 1][:-1]

                    # plan a smooth joint path from cur_qpos -> target_qpos
                    path = np.vstack((cur_qpos, now_action[:-1]))
                    try:
                        times, pos_traj, right_vel, acc, duration = planner.planner[0].TOPP(
                            path, 0.05, verbose=True
                        )
                    except Exception as e:
                        print(f"Error occurred in TOPP(single): {e}")
                        # fallback: just step once with the raw target
                        fallback = np.hstack([cur_qpos, now_action[-1]])
                        observation, reward, terminated, truncated, info = env.step(fallback)
                        last_info = info
                        continue

                    grip = now_action[-1]
                    n_step = pos_traj.shape[0]
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

                        # update DP obs after executing that mini-trajectory
                        obs_model = get_model_input(observation, [true_action], agent_num=1)
                        dp.update_obs(obs_model)

            else:
                # ================= MULTI-AGENT EXECUTION =================
                # We'll break down each agent's horizon, do TOPP per agent,
                # then step env in lockstep so both arms move together.

                # action: dict with keys like 'action_0','action_1',...
                action_dict = defaultdict(list)       # key -> [np.array(action_step)]
                action_step_dict = defaultdict(list)  # key -> [num_interpolated_steps at horizon idx]

                # First: per-agent path interpolation (TOPP)
                for aid in range(agent_num):
                    key = agent_keys[aid]

                    # collect predicted horizon for this agent
                    agent_horizon = []
                    for t in range(len(action[f'action_{aid}'])):
                        agent_horizon.append(action[f'action_{aid}'][t])

                    # up to first 8 sub-goals
                    for h in range(8):
                        now_action = agent_horizon[h]
                        raw_obs = env.get_obs()

                        # current_qpos for this agent
                        if h == 0:
                            q = raw_obs['agent'][key]['qpos'].squeeze(0)
                            q_arr = q.cpu().numpy() if hasattr(q, "cpu") else q.numpy()
                            current_qpos = q_arr[:arm_dofs_map[key]]
                        else:
                            current_qpos = agent_horizon[h - 1][:-1]

                        # plan path current_qpos -> now_action[:-1]
                        path = np.vstack((current_qpos, now_action[:-1]))
                        try:
                            times, position, right_vel, acc, duration = planner.planner[aid].TOPP(
                                path, 0.05, verbose=True
                            )
                        except Exception as e:
                            print(f"Error occurred in TOPP(multi) for agent {aid}: {e}")
                            # fallback: single step with current_qpos + gripper
                            action_now = np.hstack([current_qpos, now_action[-1]])
                            action_dict[key].append(action_now)
                            action_step_dict[key].append(1)
                            continue

                        n_step = position.shape[0]
                        action_step_dict[key].append(n_step)
                        gripper_state = now_action[-1]
                        if n_step == 0:
                            # no interpolation, just hold pose + gripper
                            action_now = np.hstack([current_qpos, gripper_state])
                            action_dict[key].append(action_now)
                        else:
                            for j in range(n_step):
                                true_action = np.hstack([position[j], gripper_state])
                                action_dict[key].append(true_action)

                # Second: execute those interpolated chunks in sync
                start_idx = [0 for _ in range(agent_num)]
                for h in range(8):
                    # figure out how many fine steps we need to run this horizon index
                    max_step = 0
                    for aid in range(agent_num):
                        key = agent_keys[aid]
                        max_step = max(
                            max_step,
                            action_step_dict[key][h] if h < len(action_step_dict[key]) else 0
                        )
                    # now interleave env.step for j in range(max_step)
                    for j in range(max_step):
                        true_action = {}
                        for aid in range(agent_num):
                            key = agent_keys[aid]
                            if h >= len(action_step_dict[key]) or len(action_dict[key]) == 0:
                                continue
                            now_step = min(j, action_step_dict[key][h] - 1)
                            true_action[key] = action_dict[key][start_idx[aid] + now_step]
                        if true_action:
                            observation, reward, terminated, truncated, info = env.step(true_action)
                    # after finishing this horizon slot h, update DP obs
                    if max_step == 0:
                        continue
                    action_concat = []
                    for aid in range(agent_num):
                        key = agent_keys[aid]
                        if h < len(action_step_dict[key]):
                            start_idx[aid] += action_step_dict[key][h]
                            if key in true_action:
                                action_concat.append(true_action[key])
                    if action_concat:
                        obs_model = get_model_input(observation, action_concat, agent_num)
                        dp.update_obs(obs_model)

            # ----- check success / render -----
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
                    print(f"[Video saved to] {record_dir}")
                print("success, step=", cnt)
                break

        # ----- per-seed logging -----
        success_rate_pct = 100.0 * total_success / total_num if total_num > 0 else 0.0
        with open(log_file, "a") as f_log:
            f_log.write(
                f"Seed={now_seed} | success={now_success} | "
                f"[Accum] success_rate={success_rate_pct:.2f}% "
                f"({total_success}/{total_num})\n"
            )

        if now_success == 0:
            print("failed")
            env.close()
        if record_dir:
            print(f"[Video saved to] {record_dir}")


if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
