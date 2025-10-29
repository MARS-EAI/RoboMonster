import time
import copy
from pathlib import Path
from typing import Callable, Optional, Union, List, Dict

import h5py
import numpy as np

from mani_skill import get_commit_info
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.io_utils import dump_json
from mani_skill.utils.structs.types import Array
from mani_skill.utils.visualization.misc import (
    images_to_video,
)
from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.utils.wrappers.record import RecordEpisode, Step, parse_env_info

from .. import sapien_utils

class RecordEpisodeMA(RecordEpisode):
    """
    Record trajectories or videos for episodes. Support multi-agent environments.
    """
    def __init__(
        self,
        env: BaseEnv,
        output_dir: str,
        save_trajectory: bool = True,
        trajectory_name: Optional[str] = None,
        save_video: bool = True,
        info_on_video: bool = False,
        save_on_reset: bool = True,
        save_video_trigger: Optional[Callable[[int], bool]] = None,
        max_steps_per_video: Optional[int] = None,
        clean_on_close: bool = True,
        record_reward: bool = True,
        record_env_state: bool = True,
        record_observation: bool = True,
        record_simple_observation: bool = False,
        video_fps: int = 30,
        avoid_overwriting_video: bool = False,
        source_type: Optional[str] = None,
        source_desc: Optional[str] = None,
    ) -> None:
        super().__init__(
            env=env,
            output_dir=output_dir,
            save_trajectory=save_trajectory,
            trajectory_name=trajectory_name,
            save_video=save_video,
            info_on_video=info_on_video,
            save_on_reset=save_on_reset,
            save_video_trigger=save_video_trigger,
            max_steps_per_video=max_steps_per_video,
            clean_on_close=clean_on_close,
            record_reward=record_reward,
            record_env_state=record_env_state,
            video_fps=video_fps,
            avoid_overwriting_video=avoid_overwriting_video,
            source_type=source_type,
            source_desc=source_desc,
        )

        self.record_observation = record_observation
        self.record_simple_observation = record_simple_observation

        if info_on_video and self.num_envs > 1:
            raise ValueError(
                "Cannot turn info_on_video=True when the number of environments parallelized is > 1"
            )

    def _target_action_dim_map(self) -> Dict[str, int]:
        """
        Returns a dict: {agent_key: target_dim}
        Priority:
          1. SuctionActionWrapper._per_agent_cfg if available
          2. Fallback to env.action_space shapes
        """
        cfg = None
        try:
            cfg = self.env.get_wrapper_attr("_per_agent_cfg")
        except Exception:
            cfg = None

        if isinstance(cfg, dict) and len(cfg) > 0:
            # has_tool_slot_ext -> +1; dofs = arm DOFs
            out = {}
            for k, v in cfg.items():
                dofs = int(v.get("dofs", 7))
                has_tool = bool(v.get("has_tool_slot_ext", False))
                out[k] = dofs + (1 if has_tool else 0)
            return out

        # Fallback: inspect action_space directly
        sp = self.env.action_space
        out = {}
        if isinstance(sp, gym_utils.spaces.Dict):
            for k, sub in sp.spaces.items():
                dim = int(np.prod(sub.shape))
                out[k] = dim
        else:
            # single agent fallback (use "__single__" as key)
            out["__single__"] = int(np.prod(sp.shape))
        return out

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
        **kwargs,
    ):
        if self.save_on_reset:
            if self.save_video and self.num_envs == 1:
                self.flush_video()
            # if doing a full reset then we flush all trajectories including incompleted ones
            if self._trajectory_buffer is not None:
                if "env_idx" not in options:
                    self.flush_trajectory(env_idxs_to_flush=np.arange(self.num_envs))
                else:
                    self.flush_trajectory(
                        env_idxs_to_flush=common.to_numpy(options["env_idx"])
                    )

        # Intentionally bypass RecordEpisode.reset; we manage first frame ourselves
        obs, info = super(RecordEpisode, self).reset(
            *args, seed=seed, options=options, **kwargs
        )
        if info["reconfigure"]:
            # if we reconfigure, there is the possibility that state dictionary looks different now
            # so trajectory buffer must be wiped
            self._trajectory_buffer = None
        if self.save_trajectory:
            state_dict = self.base_env.get_state_dict()
            dim_map = self._target_action_dim_map()
            if isinstance(self.env.action_space, gym_utils.spaces.Dict):
                # multi-agent: {agent: (1, num_envs, D)}
                action = {}
                for agent_id, D in dim_map.items():
                    action[agent_id] = np.zeros((1, self.num_envs, D), dtype=np.float32)
            else:
                # single-agent: (1, num_envs, D)
                D = dim_map.get("__single__", int(np.prod(self.env.action_space.shape)))
                action = np.zeros((1, self.num_envs, D), dtype=np.float32)
            # check if state_dict is consistent
            if not sapien_utils.is_state_dict_consistent(state_dict):
                self.record_env_state = False
                if not self._already_warned_about_state_dict_inconsistency:
                    logger.warn(
                        f"State dictionary is not consistent, disabling recording of environment states for {self.env}"
                    )
                    self._already_warned_about_state_dict_inconsistency = True
            first_step = Step(
                state=None,
                observation=common.to_numpy(common.batch(obs)),
                # we pass our dimension-correct placeholder action
                action=action,
                reward=np.zeros(
                    (
                        1,
                        self.num_envs,
                    ),
                    dtype=float,
                ),
                # terminated and truncated are fixed to be True at the start to indicate the start of an episode.
                # an episode is done when one of these is True otherwise the trajectory is incomplete / a partial episode
                terminated=np.ones((1, self.num_envs), dtype=bool),
                truncated=np.ones((1, self.num_envs), dtype=bool),
                done=np.ones((1, self.num_envs), dtype=bool),
                success=np.zeros((1, self.num_envs), dtype=bool),
                fail=np.zeros((1, self.num_envs), dtype=bool),
                env_episode_ptr=np.zeros((self.num_envs,), dtype=int),
            )
            if self.record_env_state:
                first_step.state = common.to_numpy(common.batch(state_dict))
            env_idx = np.arange(self.num_envs)
            if "env_idx" in options:
                env_idx = common.to_numpy(options["env_idx"])
            # Initialize trajectory buffer on the first episode based on given observation (which should be generated after all wrappers)
            self._trajectory_buffer = first_step
        if options is not None and "env_idx" in options:
            options["env_idx"] = common.to_numpy(options["env_idx"])
        self.last_reset_kwargs = copy.deepcopy(dict(options=options, **kwargs))
        if seed is not None:
            self.last_reset_kwargs.update(seed=seed)
        return obs, info

    def step(self, action):
        """
        Ensure recorded actions keep fixed dims over time:
          - stick: 7D
          - circle/suction/gripper: 8D
        We truncate/pad per-agent vectors to target dims (does not change env behavior).
        """
        dim_map = self._target_action_dim_map()

        def _fix_vec(vec: np.ndarray, want: int) -> np.ndarray:
            v = np.asarray(vec).reshape(-1).astype(np.float32)
            if v.size == want:
                return v
            if v.size > want:
                return v[:want]
            out = np.zeros((want,), dtype=np.float32)
            out[: v.size] = v
            return out

        if isinstance(self.env.action_space, gym_utils.spaces.Dict):
            # multi-agent dict
            assert isinstance(action, dict), "Expect dict action for multi-agent env"
            norm = {}
            keys = (
                list(self.env.action_space.spaces.keys())
                if hasattr(self.env.action_space, "spaces")
                else list(action.keys())
            )
            for k in keys:
                want = dim_map.get(k, int(np.prod(np.asarray(action[k]).shape)))
                norm[k] = _fix_vec(action[k], want)
        else:
            # single-agent
            want = dim_map.get("__single__", int(np.prod(np.asarray(action).shape)))
            norm = _fix_vec(action, want)
        # Delegate to RecordEpisode.step (this writes into buffer & forwards to env)
        return super().step(norm)

    def flush_trajectory(
        self,
        verbose=False,
        ignore_empty_transition=True,
        env_idxs_to_flush=None,
        save: bool = True,
    ):
        """
        Flushes a trajectory and by default saves it to disk

        Arguments:
            verbose (bool): whether to print out information about the flushed trajectory
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            env_idxs_to_flush: which environments by id to flush. If None, all environments are flushed.
            save (bool): whether to save the trajectory to disk
        """
        flush_count = 0
        if env_idxs_to_flush is None:
            env_idxs_to_flush = np.arange(0, self.num_envs)
        for env_idx in env_idxs_to_flush:
            start_ptr = self._trajectory_buffer.env_episode_ptr[env_idx]
            end_ptr = len(self._trajectory_buffer.done)
            if ignore_empty_transition and end_ptr - start_ptr <= 1:
                continue
            flush_count += 1
            if save:
                self._episode_id += 1
                traj_id = "traj_{}".format(self._episode_id)
                group = self._h5_file.create_group(traj_id, track_order=True)

                def recursive_add_to_h5py(
                    group: h5py.Group, data: Union[dict, Array], key
                ):
                    """simple recursive data insertion for nested data structures into h5py, optimizing for visual data as well"""
                    if isinstance(data, dict):
                        subgrp = group.create_group(key, track_order=True)
                        for k in data.keys():
                            recursive_add_to_h5py(subgrp, data[k], k)
                    else:
                        if key == "rgb":
                            # NOTE(jigu): It is more efficient to use gzip than png for a sequence of images.
                            group.create_dataset(
                                "rgb",
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                                compression="gzip",
                                compression_opts=5,
                            )
                        elif key == "depth":
                            # NOTE (stao): By default now cameras in ManiSkill return depth values of type uint16 for numpy
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                                compression="gzip",
                                compression_opts=5,
                            )
                        elif key == "seg":
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                                compression="gzip",
                                compression_opts=5,
                            )
                        elif group.name.endswith("/actions"):
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr],
                                dtype=data.dtype,
                            )
                        else:
                            group.create_dataset(
                                key,
                                data=data[start_ptr:end_ptr, env_idx],
                                dtype=data.dtype,
                            )

                # Observations need special processing
                if self.record_observation:
                    if isinstance(self._trajectory_buffer.observation, dict):
                        recursive_add_to_h5py(
                            group, self._trajectory_buffer.observation, "obs"
                        )
                    elif isinstance(self._trajectory_buffer.observation, np.ndarray):
                        if self.cpu_wrapped_env:
                            group.create_dataset(
                                "obs",
                                data=self._trajectory_buffer.observation[start_ptr:end_ptr],
                                dtype=self._trajectory_buffer.observation.dtype,
                            )
                        else:
                            group.create_dataset(
                                "obs",
                                data=self._trajectory_buffer.observation[
                                    start_ptr:end_ptr, env_idx
                                ],
                                dtype=self._trajectory_buffer.observation.dtype,
                            )
                    else:
                        raise NotImplementedError(
                            f"RecordEpisode wrapper does not know how to handle observation data of type {type(self._trajectory_buffer.observation)}"
                        )
                episode_info = dict(
                    episode_id=self._episode_id,
                    episode_seed=self.base_env._episode_seed[env_idx],
                    control_mode=self.base_env.control_mode,
                    elapsed_steps=end_ptr - start_ptr - 1,
                )
                if self.num_envs == 1:
                    episode_info.update(reset_kwargs=self.last_reset_kwargs)
                else:
                    # NOTE (stao): With multiple envs in GPU simulation, reset_kwargs do not make much sense
                    episode_info.update(reset_kwargs=dict())

                # slice some data to remove the first dummy frame.
                actions = common.index_dict_array(
                    self._trajectory_buffer.action,
                    (slice(start_ptr + 1, end_ptr), env_idx),
                )
                terminated = self._trajectory_buffer.terminated[
                    start_ptr + 1 : end_ptr, env_idx
                ]
                truncated = self._trajectory_buffer.truncated[
                    start_ptr + 1 : end_ptr, env_idx
                ]
                def _rename_agent_key_for_write(k: str) -> str:
                    if isinstance(k, str) and k.startswith("panda_stick"):
                        if "-" in k:
                            return "panda-" + k.split("-", 1)[1]
                        return "panda"
                    return k
                if isinstance(self._trajectory_buffer.action, dict):
                    actions_to_write = {}
                    for k, v in actions.items():
                        new_k = _rename_agent_key_for_write(k)
                        if (new_k in actions_to_write) and (new_k != k):
                            new_k = k
                        actions_to_write[new_k] = v
                    recursive_add_to_h5py(group, actions_to_write, "actions")
                else:
                    group.create_dataset("actions", data=actions, dtype=np.float32)
                group.create_dataset("terminated", data=terminated, dtype=bool)
                group.create_dataset("truncated", data=truncated, dtype=bool)

                if self._trajectory_buffer.success is not None:
                    end_ptr2 = len(self._trajectory_buffer.success)
                    group.create_dataset(
                        "success",
                        data=self._trajectory_buffer.success[
                            start_ptr + 1 : end_ptr2, env_idx
                        ],
                        dtype=bool,
                    )
                    episode_info.update(
                        success=self._trajectory_buffer.success[end_ptr2 - 1, env_idx]
                    )
                if self._trajectory_buffer.fail is not None:
                    group.create_dataset(
                        "fail",
                        data=self._trajectory_buffer.fail[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=bool,
                    )
                    episode_info.update(
                        fail=self._trajectory_buffer.fail[end_ptr - 1, env_idx]
                    )
                if self.record_env_state:
                    recursive_add_to_h5py(
                        group, self._trajectory_buffer.state, "env_states"
                    )
                if self.record_reward:
                    group.create_dataset(
                        "rewards",
                        data=self._trajectory_buffer.reward[
                            start_ptr + 1 : end_ptr, env_idx
                        ],
                        dtype=np.float32,
                    )

                self._json_data["episodes"].append(episode_info)
                dump_json(self._json_path, self._json_data, indent=2)
                if verbose:
                    if flush_count == 1:
                        print(f"Recorded episode {self._episode_id}")
                    else:
                        print(
                            f"Recorded episodes {self._episode_id - flush_count} to {self._episode_id}"
                        )

        # truncate self._trajectory_buffer down to save memory
        if flush_count > 0:
            self._trajectory_buffer.env_episode_ptr[env_idxs_to_flush] = (
                len(self._trajectory_buffer.done) - 1
            )
            min_env_ptr = self._trajectory_buffer.env_episode_ptr.min()
            N = len(self._trajectory_buffer.done)

            if self.record_env_state:
                self._trajectory_buffer.state = common.index_dict_array(
                    self._trajectory_buffer.state, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.observation = common.index_dict_array(
                self._trajectory_buffer.observation, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.action = common.index_dict_array(
                self._trajectory_buffer.action, slice(min_env_ptr, N)
            )
            if self.record_reward:
                self._trajectory_buffer.reward = common.index_dict_array(
                    self._trajectory_buffer.reward, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.terminated = common.index_dict_array(
                self._trajectory_buffer.terminated, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.truncated = common.index_dict_array(
                self._trajectory_buffer.truncated, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.done = common.index_dict_array(
                self._trajectory_buffer.done, slice(min_env_ptr, N)
            )
            if self._trajectory_buffer.success is not None:
                self._trajectory_buffer.success = common.index_dict_array(
                    self._trajectory_buffer.success, slice(min_env_ptr, N)
                )
            if self._trajectory_buffer.fail is not None:
                self._trajectory_buffer.fail = common.index_dict_array(
                    self._trajectory_buffer.fail, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.env_episode_ptr -= min_env_ptr

    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
        split: bool = False,
        split_width: int = 256,
    ):
        """
        Flush a video of the recorded episode(s) anb by default saves it to disk

        Arguments:
            name (str): name of the video file. If None, it will be named with the episode id.
            suffix (str): suffix to add to the video file name
            verbose (bool): whether to print out information about the flushed video
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            save (bool): whether to save the video to disk
        """
        if len(self.render_images) == 0:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            return
        if save:
            self._video_id += 1
            if name is None:
                video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
                if self._avoid_overwriting_video:
                    while (
                        Path(self.output_dir)
                        / (video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
                    ).exists():
                        self._video_id += 1
                        video_name = "{}".format(self._video_id)
                        if suffix:
                            video_name += "_" + suffix
            else:
                video_name = name
            if split:
                height, width, _ = self.render_images[0].shape
                for i in range(0, width, split_width):
                    split_images = []
                    for j in range(0, len(self.render_images)):
                        split_image = self.render_images[j][:, i : i + split_width, :]
                        split_images.append(split_image)
                    import os
                    output_dir = os.path.join(self.output_dir, video_name)
                    images_to_video(
                        split_images,
                        str(output_dir),
                        video_name=video_name + f"_{(int)(i / height)}",
                        fps=self.video_fps,
                        verbose=verbose,
                    )
            else:
                images_to_video(
                    self.render_images,
                    str(self.output_dir),
                    video_name=video_name,
                    fps=self.video_fps,
                    verbose=verbose,
                )
        self._video_steps = 0
        self.render_images = []