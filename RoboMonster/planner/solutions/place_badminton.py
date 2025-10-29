import numpy as np
import sapien
import time

from tasks import PlaceBadmintonEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: PlaceBadmintonEnv, seed=None, debug=False, vis=False, tool_modes=None):
    env.reset(seed=seed)
    if tool_modes is None:
        tool_modes = ["gripper", "gripper"]
    base_pose = [ag.robot.pose for ag in env.agent.agents]
    is_multi_agent = True

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base_pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        is_multi_agent=is_multi_agent,
        tool_modes=tool_modes,
    )

    pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.badminton, actor_data=env.annotation_data['badminton'], pre_dis=0, id=0)
    pose1[2] += 0.2
    planner.move_to_pose_with_screw(pose=pose1, move_id=0)
    pose1[2] -= 0.34
    planner.move_to_pose_with_screw(pose=pose1, move_id=0)

    # Stick
    if tool_modes == ["gripper", "stick"]:
        planner.close_gripper(close_id=0)

    # Gripper
    if tool_modes == ["gripper", "gripper"]:
        planner.close_gripper(close_id=[0,1])

    pose1[2] += 0.2
    planner.move_to_pose_with_screw(pose=pose1, move_id=0)
    pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.barrel, actor_data=env.annotation_data['barrel'], pre_dis=0, id=0)
    planner.move_to_pose_with_screw(pose=pose2, move_id=0)
    pose2[2] += 0.01
    pose2[0] -= 0.085
    planner.move_to_pose_with_screw(pose=pose2, move_id=0)
    planner.open_gripper(open_id=0)
    pose2[0] += 0.1
    planner.move_to_pose_with_screw(pose=pose2, move_id=0)
    planner.move_to_pose_with_screw(pose=pose1, move_id=0)
    print(env.badminton.pose.p)
    print(env.barrel.pose.p)
    pose3 = planner.get_target_pose_w_labeled_direction(actor=env.barrel, actor_data=env.annotation_data['barrel'], pre_dis=0, id=0)
    pose3[0] += 0.1
    planner.move_to_pose_with_screw(pose=pose3, move_id=1)
    if tool_modes == ["gripper", "stick"]:
        pose3[0] -= 0.28
    else:
        pose3[0] -= 0.23
    res = planner.move_to_pose_with_screw(pose=pose3, move_id=1)
    print(env.badminton.pose.p)
    print(env.barrel.pose.p)
    # res = {'success': True}
    planner.close()
    return res
