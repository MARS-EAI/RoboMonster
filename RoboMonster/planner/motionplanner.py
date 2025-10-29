# planner/motionplanner.py
import numpy as np
import sapien
import sapien.physx as physx

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import to_sapien_pose

from utils.mplib_utils import FlexiblePlanner
from .utils import get_actor_obb, compute_grasp_info_by_obb, build_panda_gripper_grasp_pose_visual

from transforms3d.euler import quat2euler, euler2quat
from copy import deepcopy
import transforms3d as t3d
from typing import Union, List, Any, Optional

from gymnasium import spaces as gym_spaces
from utils.suction.suction import SuctionTool


class PandaArmMotionPlanningSolver:

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: Union[sapien.Pose, List[sapien.Pose]] = None,
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits=0.9,
        joint_acc_limits=0.9,
        is_multi_agent: bool = False,
        tool_modes: Union[str, List[str]] = None,   # ("gripper" | "suction" | "stick" | "circle")
    ):
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.is_multi_agent = is_multi_agent

        if self.is_multi_agent:
            assert hasattr(self.base_env.agent, "agents"), "Multi-agent environment must have agents attribute"
            self.env_agent: List[BaseAgent] = self.base_env.agent.agents
        else:
            self.env_agent: List[BaseAgent] = [self.base_env.agent]

        self.arm_dofs = []
        for ag in self.env_agent:
            name = self._get_robot_name(ag)
            if name in ("panda", "panda_stick", "panda_circle"):
                self.arm_dofs.append(7)
            elif name == "xarm6_robotiq":
                self.arm_dofs.append(6)
            else:
                self.arm_dofs.append(len(ag.robot.get_active_joints()) - 1)

        self.robot_name = self._get_robot_name(self.env_agent[0])
        if self.robot_name in ("panda", "panda_stick", "panda_circle"):
            self.OPEN, self.CLOSED = 1, -1
            self.eef_name = "panda_hand_tcp"
            self.tcp_offset = np.zeros(3)
        elif self.robot_name == "xarm6_robotiq":
            self.OPEN, self.CLOSED = 0, 0.8
            self.eef_name = "eef"
            self.tcp_offset = np.array([0, 0, -0.015])
        else:
            self.OPEN, self.CLOSED = 1, -1
            self.eef_name = "ee"
            self.tcp_offset = np.zeros(3)

        self.agent_num = len(self.env_agent) if is_multi_agent else 1
        self.robot = [agent.robot for agent in self.env_agent]
        self.base_pose = [base_pose] if not isinstance(base_pose, list) else base_pose
        self.control_mode = self.env_agent[0].control_mode
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        print("Robot Control Mode:", self.control_mode)

        self.planner = self.setup_planner()

        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            self.grasp_pose_visual = []
            for i in range(self.agent_num):
                vis_actor = build_panda_gripper_grasp_pose_visual(self.base_env.scene, "grasp_pose_visual" + str(i))
                self.grasp_pose_visual.append(vis_actor)
                self.grasp_pose_visual[i].set_pose(self.base_pose[i])

        self.elapsed_steps = 0
        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

        for ag in self.env_agent:
            if not hasattr(ag, "robot_uid"):
                ag.robot_uid = ag.uid
        if isinstance(self.base_env.action_space, gym_spaces.Dict):
            self.agent_keys = list(self.base_env.action_space.spaces.keys())
        else:
            self.agent_keys = ["agent"]
        if isinstance(tool_modes, str):
            assert tool_modes in ("gripper", "suction", "stick", "circle")
            self.tool_modes = [tool_modes] * self.agent_num
        else:
            assert len(tool_modes) == self.agent_num, "tool_modes length must match the number of agents"
            for m in tool_modes:
                assert m in ("gripper", "suction", "stick", "circle")
            self.tool_modes = list(tool_modes)

        self.no_tool_slot = [m == "stick" for m in self.tool_modes]

        self.tool_cmd = []
        for aid in range(self.agent_num):
            m = self.tool_modes[aid]
            if m == "gripper":
                self.tool_cmd.append(float(self.OPEN))
            elif m == "suction":
                self.tool_cmd.append(-1)   # suction default: close(-0.5)
            elif m == "circle":
                self.tool_cmd.append(+2.0)   # circle default: open(+2)
            else:  # stick
                self.tool_cmd.append(None)

        self.gripper_state = [self.OPEN] * self.agent_num

        self.suction = None
        if any(m == "suction" for m in self.tool_modes):
            shared = getattr(self.base_env, "_shared_suction_tool", None)
            if shared is not None:
                self.suction = shared
            else:
                self.suction = SuctionTool(env=self.base_env, debug=True)

            try:
                if not self.suction.has_fingers():
                    finger_links = []
                    for ag in self.env_agent:
                        for l in ag.robot.get_links():
                            nm = l.get_name() if hasattr(l, "get_name") else getattr(l, "name", "")
                            low = nm.lower()
                            if ("finger" in low) or ("pad" in low):
                                finger_links.append(l)
                    if finger_links:
                        self.suction.set_fingers(finger_links)
            except Exception:
                pass

    def _attached_from_info(self, info, agent_id=0):
        if not isinstance(info, dict):
            return False
        if isinstance(self.agent_keys, list) and len(self.agent_keys) > 1:
            key = self.agent_keys[agent_id]
            return bool(info.get(key, {}).get("suction_attached", False))
        return bool(info.get("suction_attached", False))

    def _action_hold_here(self, agent_id: int, flag: Optional[float]):
        slot = float(flag) if (flag is not None) else 0.0
        if self.control_mode == "pd_joint_pos":
            q = self.robot[agent_id].get_qpos()[0, : self.arm_dofs[agent_id]].cpu().numpy()
            return np.hstack([q, slot]).astype(np.float32)
        elif self.control_mode == "pd_joint_pos_vel":
            q = self.robot[agent_id].get_qpos()[0, : self.arm_dofs[agent_id]].cpu().numpy()
            v = np.zeros_like(q, dtype=np.float32)
            return np.hstack([q, v, slot]).astype(np.float32)
        elif self.control_mode == "pd_ee_pose":
            ee_pose_arr = self.get_current_ee_pose(agent_id=agent_id)
            return np.hstack([ee_pose_arr, slot]).reshape(1, -1).astype(np.float32)
        else:
            raise NotImplementedError(f"_action_hold_here CAN'T SUPPORT control_mode: {self.control_mode}")

    def open_circle(self, probe_steps: int = 2, agent_id: int = 0):
        assert self.tool_modes[agent_id] == "circle", "this agent ISN'T circle"
        obs = reward = terminated = truncated = info = None
        for _ in range(max(1, int(probe_steps))):
            if not self.is_multi_agent:
                action = self._action_hold_here(agent_id, +2.0)
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                adict = {}
                for aid in range(self.agent_num):
                    flag = (+2.0 if aid == agent_id and not self.no_tool_slot[aid] else self.tool_cmd[aid])
                    adict[self.agent_keys[aid]] = self._action_hold_here(aid, flag)
                obs, reward, terminated, truncated, info = self.env.step(adict)
            if self.vis:
                self.base_env.render_human()
        self.tool_cmd[agent_id] = +2.0
        return obs, reward, terminated, truncated, info

    def close_circle(self, settle_steps: int = 2, agent_id: int = 0):
        assert self.tool_modes[agent_id] == "circle", "this agent ISN'T circle"
        obs = reward = terminated = truncated = info = None
        for _ in range(max(1, int(settle_steps))):
            if not self.is_multi_agent:
                action = self._action_hold_here(agent_id, -2.0)
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                adict = {}
                for aid in range(self.agent_num):
                    flag = (-2.0 if aid == agent_id and not self.no_tool_slot[aid] else self.tool_cmd[aid])
                    adict[self.agent_keys[aid]] = self._action_hold_here(aid, flag)
                obs, reward, terminated, truncated, info = self.env.step(adict)
            if self.vis:
                self.base_env.render_human()
        self.tool_cmd[agent_id] = -2.0
        return obs, reward, terminated, truncated, info

    def open_suction(self, agent_id: int = 0):
        assert self.tool_modes[agent_id] == "suction", "this agent ISN'T suction"
        assert self.suction is not None, "suction DIDN'T INIT"
        target_value = +1.0
        attached = False
        obs = reward = terminated = truncated = info = None
        steps = 20 if self.is_multi_agent else 1
        for _ in range(steps):
            if self.is_multi_agent:
                self.tool_cmd[agent_id] = min(self.tool_cmd[agent_id] + 0.1, target_value)
            else:
                self.tool_cmd[agent_id] = target_value

            if not self.is_multi_agent:
                action = self._action_hold_here(agent_id, self.tool_cmd[agent_id])
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                adict = {}
                for aid in range(self.agent_num):
                    if aid == agent_id:
                        flag = self.tool_cmd[aid]
                    else:
                        flag = None if self.no_tool_slot[aid] else self.tool_cmd[aid]
                    adict[self.agent_keys[aid]] = self._action_hold_here(aid, flag)
                obs, reward, terminated, truncated, info = self.env.step(adict)
            if self.vis:
                self.base_env.render_human()
            attached = self._attached_from_info(info, agent_id)
            if attached:
                break
        return attached, obs, reward, terminated, truncated, info

    def close_suction(self, agent_id: int = 0):
        assert self.tool_modes[agent_id] == "suction", "this agent ISN'T suction"
        assert self.suction is not None, "suction DIDN'T INIT"
        target_value = -1.0
        obs = reward = terminated = truncated = info = None
        steps = 20 if self.is_multi_agent else 1
        for _ in range(steps):
            if self.is_multi_agent:
                self.tool_cmd[agent_id] = max(self.tool_cmd[agent_id] - 0.1, target_value)
            else:
                self.tool_cmd[agent_id] = target_value

            if not self.is_multi_agent:
                action = self._action_hold_here(agent_id, self.tool_cmd[agent_id])
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                adict = {}
                for aid in range(self.agent_num):
                    if aid == agent_id:
                        flag = self.tool_cmd[aid]
                    else:
                        flag = None if self.no_tool_slot[aid] else self.tool_cmd[aid]
                    adict[self.agent_keys[aid]] = self._action_hold_here(aid, flag)
                obs, reward, terminated, truncated, info = self.env.step(adict)
            if self.vis:
                self.base_env.render_human()
        self.tool_cmd[agent_id] = target_value
        return True, obs, reward, terminated, truncated, info

    # ======== Utility functions ========
    def _get_robot_name(self, agent):
        if hasattr(agent, "uid"):
            return agent.uid.split("-", 1)[0]
        if hasattr(agent, "robot_uid"):
            return agent.robot_uid.split("-", 1)[0]
        name = agent.__class__.__name__.lower()
        if "panda" in name:
            return "panda"
        if "xarm" in name:
            return "xarm6_robotiq"
        return name

    def render_wait(self):
        if not self.vis or not self.debug:
            return
        print("Press [c] to continue")
        viewer = self.base_env.render_human()
        while True:
            if viewer.window.key_down("c"):
                break
            self.base_env.render_human()

    def setup_planner(self):
        planner_group = []
        for i in range(self.agent_num):
            link_names = [link.get_name() for link in self.robot[i].get_links()]
            joint_names = [joint.get_name() for joint in self.robot[i].get_active_joints()]
            if self.robot_name == "panda":
                joint_vel_limits = np.ones(7) * self.joint_vel_limits
                joint_acc_limits = np.ones(7) * self.joint_acc_limits
            elif self.robot_name == "xarm6_robotiq":
                joint_vel_limits = np.ones(6) * self.joint_vel_limits
                joint_acc_limits = np.ones(6) * self.joint_acc_limits
            else:
                dof = len(joint_names)
                joint_vel_limits = np.ones(dof) * self.joint_vel_limits
                joint_acc_limits = np.ones(dof) * self.joint_acc_limits

            planner = FlexiblePlanner(
                urdf=self.env_agent[i].urdf_path,
                srdf=self.env_agent[i].urdf_path.replace(".urdf", ".srdf"),
                user_link_names=link_names,
                user_joint_names=joint_names,
                move_group=self.eef_name,
                joint_vel_limits=joint_vel_limits,
                joint_acc_limits=joint_acc_limits,
            )
            self.base_pose[i] = to_sapien_pose(self.base_pose[i])
            planner.set_base_pose(np.hstack([self.base_pose[i].p, self.base_pose[i].q]))
            planner_group.append(planner)
        return planner_group

    def convert_joint_to_ee(self, joint_trajectory: np.ndarray, agent_id: int = 0) -> np.ndarray:
        """
        Converts a joint space trajectory to an end-effector pose trajectory.
        
        Args:
            joint_trajectory (np.ndarray): The joint position trajectory, shape (N, D_joint).
            agent_id (int): The ID of the agent/planner to use for the conversion.

        Returns:
            np.ndarray: The end-effector pose trajectory, shape (N, 6), where each row is [px, py, pz, roll, pitch, yaw].
        """
        planner = self.planner[agent_id]
        ee_trajectory = []
        pinocchio_model = planner.pinocchio_model
        move_group_link_id = planner.move_group_link_id
        full_qpos = planner.robot.get_qpos() 

        for qpos_step in joint_trajectory:
            full_qpos[:len(qpos_step)] = qpos_step
            pinocchio_model.compute_forward_kinematics(full_qpos)
            ee_pose_or_array = pinocchio_model.get_link_pose(move_group_link_id)
            if isinstance(ee_pose_or_array, sapien.Pose):
                pos = ee_pose_or_array.p
                quat = ee_pose_or_array.q
            elif isinstance(ee_pose_or_array, np.ndarray):
                pos = ee_pose_or_array[:3]
                quat = ee_pose_or_array[3:]
            else:
                raise TypeError(f"Unexpected type returned by get_link_pose: {type(ee_pose_or_array)}")
            rpy = quat2euler(quat, axes='rxyz')
            ee_pose_np = np.hstack([pos, rpy])  # [px, py, pz, roll, pitch, yaw]
            ee_trajectory.append(ee_pose_np)
        return np.array(ee_trajectory)

    def get_current_ee_pose(self, agent_id: int = 0) -> np.ndarray:
        """
        From current robot joints pose, calculate EE pose: numpy array [px, py, pz, qw, qx, qy, qz]
        """
        # get qpos
        full_qpos = self.robot[agent_id].get_qpos().cpu().numpy()[0]
        arm_joints = full_qpos[:7]
        ee_traj = self.convert_joint_to_ee(np.array([arm_joints]), agent_id=agent_id)
        return ee_traj[0] 

    def follow_path(self, result_group, move_id, refine_steps: int = 0):
        n_step = 0
        for r in result_group:
            n_step = max(n_step, r["position"].shape[0])

        planned_ee_trajectories = {}
        if self.control_mode == "pd_ee_pose":
            for idx, agent_id in enumerate(move_id):
                joint_traj = result_group[idx]["position"]
                ee_traj = self.convert_joint_to_ee(joint_traj, agent_id=agent_id)
                planned_ee_trajectories[agent_id] = ee_traj

        for step in range(n_step + refine_steps):

            if self.suction is not None:
                self.suction.update_attachment()

            if not self.is_multi_agent:
                agent_id = move_id[0]
                slot = float(self.tool_cmd[agent_id]) if (self.tool_cmd[agent_id] is not None) else 0.0
                if self.control_mode == "pd_ee_pose":
                    ee_traj = planned_ee_trajectories[agent_id]
                    idx = min(step, len(ee_traj) - 1)
                    target_ee_pose = ee_traj[idx]
                    action = np.hstack([target_ee_pose, slot]).reshape(1, -1).astype(np.float32)
                else:
                    if self.control_mode == "pd_joint_pos_vel":
                        qpos = result_group[0]["position"][min(step, n_step - 1)]
                        qvel = result_group[0]["velocity"][min(step, n_step - 1)]
                        action = np.hstack([qpos, qvel, slot]).astype(np.float32)
                    else:
                        qpos = result_group[0]["position"][min(step, n_step - 1)]
                        action = np.hstack([qpos, slot]).astype(np.float32)
                obs, reward, terminated, truncated, info = self.env.step(action)

            else:
                action_dict = {}
                for aid in range(self.agent_num):
                    slot = float(self.tool_cmd[aid]) if (self.tool_cmd[aid] is not None) else 0.0
                    if self.control_mode == "pd_ee_pose":
                        if aid not in move_id:
                            ee_pose_arr = self.get_current_ee_pose(agent_id=aid)
                            action = np.hstack([ee_pose_arr, slot]).reshape(1, -1).astype(np.float32)
                        else:
                            ee_traj = planned_ee_trajectories[aid]
                            idx = min(step, len(ee_traj) - 1)
                            target_ee_pose = ee_traj[idx]
                            action = np.hstack([target_ee_pose, slot]).reshape(1, -1).astype(np.float32)
                    else:
                        if aid not in move_id:
                            qpos = self.robot[aid].get_qpos()[0, : self.arm_dofs[aid]].cpu().numpy()
                            if self.control_mode == "pd_joint_pos_vel":
                                action = np.hstack([qpos, qpos * 0, slot]).astype(np.float32)
                            else:
                                action = np.hstack([qpos, slot]).astype(np.float32)
                        else:
                            move_idx = move_id.index(aid)
                            self_step = result_group[move_idx]["position"].shape[0]
                            qpos = result_group[move_idx]["position"][min(step, self_step - 1)]
                            if self.control_mode == "pd_joint_pos_vel":
                                qvel = result_group[move_idx]["velocity"][min(step, self_step - 1)]
                                action = np.hstack([qpos, qvel, slot]).astype(np.float32)
                            else:
                                action = np.hstack([qpos, slot]).astype(np.float32)
                    action_dict[self.agent_keys[aid]] = action
                obs, reward, terminated, truncated, info = self.env.step(action_dict)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
            if self.vis:
                self.base_env.render_human()
        return True, obs, reward, terminated, truncated, info

    def _hold_current_pose(self, steps: int = 1, agent_id: int = 0):
        obs = reward = terminated = truncated = info = None
        for _ in range(steps):
            if self.suction is not None:
                self.suction.update_attachment()
            if not self.is_multi_agent:
                action = self._action_hold_here(agent_id=agent_id, flag=self.tool_cmd[agent_id])
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                action_dict = {}
                for aid in range(self.agent_num):
                    action = self._action_hold_here(agent_id=aid, flag=self.tool_cmd[aid])
                    action_dict[self.agent_keys[aid]] = action
                obs, reward, terminated, truncated, info = self.env.step(action_dict)
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def move_to_pose_with_screw(
        self, pose: Union[Any, List[Any]], dry_run: bool = False, refine_steps: int = 0, move_id : Union[int, List[int]] = 0, jump: int = 1
    ):
        pose = [pose, ] if not isinstance(pose, list) else pose
        pose = [to_sapien_pose(p) for p in pose]
        for i in range(len(pose)):
            p = np.array(pose[i].p) + self.tcp_offset
            pose[i] = sapien.Pose(p, pose[i].q)
        move_id = [move_id, ] if not isinstance(move_id, list) else move_id
        # try screw two times before giving up
        # if self.grasp_pose_visual is not None:
        #     for id in range(len(pose)):
        #         self.grasp_pose_visual[move_id[id]].set_pose(pose[id])
        #     if self.vis:
        #         self.base_env.render_human()
        result_group = []
        for id in range(len(pose)):
            planner_id = move_id[id]
            result = self.planner[planner_id].plan_screw(
                np.concatenate([pose[id].p, pose[id].q]),
                self.robot[planner_id].get_qpos().cpu().numpy()[0],
                time_step=self.base_env.control_timestep,
                use_point_cloud=self.use_point_cloud,
            )
            if result["status"] != "Success":
                result = self.planner[planner_id].plan_screw(
                    np.concatenate([pose[id].p, pose[id].q]),
                    self.robot[planner_id].get_qpos().cpu().numpy()[0],
                    time_step=self.base_env.control_timestep,
                    use_point_cloud=self.use_point_cloud,
                )
                if result["status"] != "Success":
                    print("Failed to plan screw motion in agent ", id)
                    self.render_wait()
                    return -1
            self.render_wait()
            if dry_run:
                return result
            result_group.append(result)
        return self.follow_path(result_group, move_id, refine_steps=refine_steps)

    def open_gripper(self, open_id: Union[int, List[int]] = 0):
        ids = [open_id] if not isinstance(open_id, list) else open_id
        ids = [aid for aid in ids if self.tool_modes[aid] == "gripper"]
        if not ids:
            return None, None, None, None, None

        for aid in ids:
            self.gripper_state[aid] = self.OPEN
            self.tool_cmd[aid] = float(self.OPEN)

        for _ in range(20):
            if not self.is_multi_agent:
                if self.control_mode == "pd_ee_pose":
                    ee_pose_arr = self.get_current_ee_pose(agent_id=0)
                    action = np.hstack([ee_pose_arr, self.tool_cmd[0]]).reshape(1, -1)
                else:
                    if self.tool_modes[0] == "gripper":
                        self.tool_cmd[0] = min(self.tool_cmd[0] + 0.1, float(self.OPEN))
                    qpos = self.robot[0].get_qpos()[0, : self.arm_dofs[0]].cpu().numpy()
                    if self.control_mode == "pd_joint_pos":
                        action = np.hstack([qpos, self.tool_cmd[0]])
                    else:
                        action = np.hstack([qpos, qpos * 0, self.tool_cmd[0]])
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                action_dict = {}
                for aid in range(self.agent_num):
                    if self.control_mode == "pd_ee_pose":
                        ee_pose_arr = self.get_current_ee_pose(agent_id=aid)
                        action = np.hstack([ee_pose_arr, self.tool_cmd[aid]]).reshape(1, -1)
                    else:
                        qpos = self.robot[aid].get_qpos()[0, : self.arm_dofs[aid]].cpu().numpy()
                        if self.control_mode == "pd_joint_pos":
                            action = np.hstack([qpos, self.tool_cmd[aid]]) if not self.no_tool_slot[aid] else qpos
                        else:
                            action = np.hstack([qpos, qpos * 0, self.tool_cmd[aid]]) if not self.no_tool_slot[aid] else np.hstack([qpos, qpos * 0])
                    if self.tool_modes[aid] == "gripper" and aid in ids:
                        self.tool_cmd[aid] = min(self.tool_cmd[aid] + 0.1, float(self.OPEN))
                        if not self.no_tool_slot[aid]:
                            action[-1] = self.tool_cmd[aid]
                    action_dict[self.agent_keys[aid]] = action
                obs, reward, terminated, truncated, info = self.env.step(action_dict)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Open gripper step: reward={reward} info={info}")
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, close_id: Union[int, List[int]] = 0):
        ids = [close_id] if not isinstance(close_id, list) else close_id
        ids = [aid for aid in ids if self.tool_modes[aid] == "gripper"]
        if not ids:
            return None, None, None, None, None

        for aid in ids:
            self.gripper_state[aid] = self.CLOSED
            self.tool_cmd[aid] = float(self.CLOSED)

        for _ in range(20):
            if not self.is_multi_agent:
                if self.control_mode == "pd_ee_pose":
                    ee_pose_arr = self.get_current_ee_pose(agent_id=0)
                    action = np.hstack([ee_pose_arr, self.tool_cmd[0]]).reshape(1, -1)
                else:
                    if self.tool_modes[0] == "gripper":
                        self.tool_cmd[0] = max(self.tool_cmd[0] - 0.1, float(self.CLOSED))
                    qpos = self.robot[0].get_qpos()[0, : self.arm_dofs[0]].cpu().numpy()
                    if self.control_mode == "pd_joint_pos":
                        action = np.hstack([qpos, self.tool_cmd[0]])
                    else:
                        action = np.hstack([qpos, qpos * 0, self.tool_cmd[0]])
                obs, reward, terminated, truncated, info = self.env.step(action)
            else:
                action_dict = {}
                for aid in range(self.agent_num):
                    if self.control_mode == "pd_ee_pose":
                        ee_pose_arr = self.get_current_ee_pose(agent_id=aid)
                        action = np.hstack([ee_pose_arr, self.tool_cmd[aid]]).reshape(1, -1)
                    else:
                        qpos = self.robot[aid].get_qpos()[0, : self.arm_dofs[aid]].cpu().numpy()
                        if self.control_mode == "pd_joint_pos":
                            action = np.hstack([qpos, self.tool_cmd[aid]]) if not self.no_tool_slot[aid] else qpos
                        else:
                            action = np.hstack([qpos, qpos * 0, self.tool_cmd[aid]]) if not self.no_tool_slot[aid] else np.hstack([qpos, qpos * 0])
                    if self.tool_modes[aid] == "gripper" and aid in ids:
                        self.tool_cmd[aid] = max(self.tool_cmd[aid] - 0.1, float(self.CLOSED))
                        if not self.no_tool_slot[aid]:
                            action[-1] = self.tool_cmd[aid]
                    action_dict[self.agent_keys[aid]] = action
                obs, reward, terminated, truncated, info = self.env.step(action_dict)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(f"[{self.elapsed_steps:3}] Close gripper step: reward={reward} info={info}")
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    """  
        The following method is revised from RoboTwin.  
        It computes the grasp pose by pre-marking the contact points,  
        functional points, etc. and their directions on the object.  
    """  
    def get_target_pose_w_labeled_direction(self, actor, actor_data, pre_dis = 0.0, id = 0):
        actor_matrix = actor.pose.to_transformation_matrix()
        actor_matrix = actor_matrix[0]
        local_target_matrix = np.asarray(actor_data['target_pose'][id])
        local_target_matrix[:3,3] *= actor_data['scale']
        convert_matrix = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
        global_target_pose_matrix = actor_matrix.cpu().numpy() @ local_target_matrix @ convert_matrix
        global_target_pose_matrix_q = global_target_pose_matrix[:3,:3]
        global_target_pose_p = global_target_pose_matrix[:3,3] + global_target_pose_matrix_q @ np.array([pre_dis,0,0]).T
        global_target_pose_q = t3d.quaternions.mat2quat(global_target_pose_matrix_q)
        res_pose = list(global_target_pose_p)+list(global_target_pose_q)
        return np.array(res_pose)
        
    def get_grasp_pose_w_labeled_direction(self, actor, actor_data, pre_dis = 0.0, id = 0):
        actor_matrix = actor.pose.to_transformation_matrix()
        actor_matrix = actor_matrix[0]
        local_contact_matrix = np.asarray(actor_data['contact_points_pose'][id])
        local_contact_matrix[:3,3] *= actor_data['scale']
        convert_matrix = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
        global_contact_pose_matrix = actor_matrix.cpu().numpy() @ local_contact_matrix @ convert_matrix
        global_contact_pose_matrix_q = global_contact_pose_matrix[:3,:3]
        global_grasp_pose_p = global_contact_pose_matrix[:3,3] + global_contact_pose_matrix_q @ np.array([pre_dis,0,0]).T
        global_grasp_pose_q = t3d.quaternions.mat2quat(global_contact_pose_matrix_q)
        res_pose = list(global_grasp_pose_p)+list(global_grasp_pose_q)
        return np.array(res_pose)
    
    def is_plan_success(self, grasp_pose: list, robot_id : int = 0):
        las_robot_qpose = self.planner[robot_id].robot.get_qpos()
        result = self.planner[robot_id].plan_screw(
            target_pose=grasp_pose,
            qpos=self.robot[robot_id].get_qpos().cpu().numpy()[0],
            time_step=1 / 250,
            use_point_cloud=False,
            use_attach=False,
        )
        self.planner[robot_id].robot.set_qpos(las_robot_qpose,full=True)
        return result["status"] == "Success" and result["position"].shape[0] <= 2000

    def evaluate_grasp_pose(self, endpose, grasp_pose: list, agent_id : int = 0):
        is_plan_suc = self.is_plan_success(grasp_pose=grasp_pose, robot_id=agent_id)
        if not is_plan_suc:
            return -1e10
    
        res = 0
        res += np.sqrt(np.sum((np.array(endpose.p[0]) - np.array(grasp_pose)[:3]) ** 2)) / 0.7
        trans_now_pose_matrix = t3d.quaternions.quat2mat(grasp_pose[3:]) @ np.linalg.inv(endpose.to_transformation_matrix()[0][:3,:3])
        theta_xy = np.mod(np.abs(t3d.euler.mat2euler(trans_now_pose_matrix)[:2]), np.pi)
        res += 2 * np.sum(theta_xy/np.pi)
        return -res
      
    def get_grasp_pose_from_goal_point_and_direction(self, actor, actor_data, endpose, actor_functional_point_id = 0, target_point = None,
                                                     target_approach_direction = [0,0,1,0], pre_dis = 0.):
        target_approach_direction_mat = t3d.quaternions.quat2mat(target_approach_direction)
        actor_matrix = actor.pose.to_transformation_matrix()[0]
        if type(target_point) == np.ndarray:
            target_point_copy = target_point
        else:
            target_point_copy = deepcopy(target_point.raw_pose[0][:3])
        target_point_copy -= target_approach_direction_mat @ np.array([0,0,pre_dis])
        adjunction_matrix_list = [
            # 90 degree
            t3d.euler.euler2mat(0,0,0),
            t3d.euler.euler2mat(0,0,np.pi/2),
            t3d.euler.euler2mat(0,0,-np.pi/2),
            t3d.euler.euler2mat(0,0,np.pi),
            # 45 degree
            t3d.euler.euler2mat(0,0,np.pi/4),
            t3d.euler.euler2mat(0,0,np.pi*3/4),
            t3d.euler.euler2mat(0,0,-np.pi*3/4),
            t3d.euler.euler2mat(0,0,-np.pi/4),
        ]

        res_pose = None
        res_eval= -1e10
        for adjunction_matrix in adjunction_matrix_list:
            local_target_matrix = np.asarray(actor_data['functional_matrix'][actor_functional_point_id])
            local_target_matrix[:3,3] *= actor_data['scale']
            fuctional_matrix = actor_matrix[:3,:3] @ np.asarray(actor_data['functional_matrix'][actor_functional_point_id])[:3,:3]
            fuctional_matrix = fuctional_matrix @ adjunction_matrix
            trans_matrix = target_approach_direction_mat @ np.linalg.inv(fuctional_matrix)
            end_effector_pose_matrix = t3d.quaternions.quat2mat(endpose.q[0]) @ np.array([[1,0,0],[0,1,0],[0,0,1]])
            target_grasp_matrix = trans_matrix @ end_effector_pose_matrix
         
            res_matrix = np.eye(4)
            res_matrix[:3,3] = (actor_matrix  @ local_target_matrix)[:3,3] - endpose.p[0]
            res_matrix[:3,3] = np.linalg.inv(end_effector_pose_matrix) @ res_matrix[:3,3]
            target_grasp_qpose = t3d.quaternions.mat2quat(target_grasp_matrix)
            now_pose = (target_point_copy - target_grasp_matrix @ res_matrix[:3,3]).tolist() + target_grasp_qpose.tolist()
            now_pose_eval = self.evaluate_grasp_pose(endpose, now_pose)
            if not self.is_plan_success(now_pose):
                continue
            elif now_pose_eval > res_eval:
                res_pose = now_pose
                res_eval = now_pose_eval
            else:
                res_pose = now_pose
        if res_pose is None:
            return None
        return np.array(res_pose)

    """  
        The following method is revised from the official ManiSkill code. 
        It constructs an OBB (Oriented Bounding Box) for the object  
        and computes the grasp pose based on approaching, closing, and center.  
    """ 
    def get_grasp_pose_from_obb(self, actor, agent_id : int = 0):
        obb = get_actor_obb(actor)
        approaching = np.array([0, 0, -1])
        target_closing = self.env_agent[agent_id].tcp.pose.to_transformation_matrix()[0, :3, 1].numpy()
        FINGER_LENGTH = 0.02 + 0.005
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = self.env_agent[agent_id].build_grasp_pose(approaching, closing, center)
        # search for a valid grasp pose
        angles = np.arange(0, 2 * np.pi / 3, np.pi / 2)
        now_res = -1e11
        endpose = self.env_agent[agent_id].tcp.pose
        for angle in angles:
            delta_pose = sapien.Pose(q=euler2quat(0, 0, angle))
            grasp_pose2 = grasp_pose * delta_pose
            grasp_list = list(grasp_pose2.p) + list(grasp_pose2.q)
            is_plan_suc = self.is_plan_success(grasp_pose=grasp_list, robot_id=agent_id)
            if not is_plan_suc:
                continue
            else:
                grasp_pose = grasp_pose2
                now_res = 0
                break
        if now_res < -1e10:
            raise RuntimeError(f"Failed to find a valid grasp pose for {actor.name}")
        return np.array(list(grasp_pose.p) + list(grasp_pose.q))
    
    def get_grasp_pose_for_stack(self, now_pose, target_actor, height_offset=0.05):
        target_pose = target_actor.pose
        target_pose = target_pose * sapien.Pose([0, 0, height_offset])
        offset = (target_pose.p - now_pose[:3]).numpy()[0]
        align_pose = sapien.Pose(now_pose[:3] + offset, now_pose[3:])
        return np.array(list(align_pose.p) + list(align_pose.q))

    def close(self):
        pass
