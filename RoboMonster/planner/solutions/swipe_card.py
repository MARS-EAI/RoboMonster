import numpy as np
import sapien
import time

from tasks import SwipeCardEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: SwipeCardEnv, seed=None, debug=False, vis=False, tool_modes=None):
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

    # Suction
    if tool_modes == ["suction", "gripper"]:
        tool = getattr(env.unwrapped, "_shared_suction_tool", None)
        assert tool is not None, "SuctionTool not found; make sure SuctionActionWrapper is applied."
        tool.clear_filters()
        tool.allow_only_names(contains=["card"])          
        tool.disallow_names(contains=["table", "floor", "ground", "cube"])
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.card, actor_data=env.annotation_data['card'], pre_dis=0, id=0)
        pose1[2] += 0.2
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        pose1[2] -= 0.2

        # RoboCasa!
        pose1[2] += 0.02
        
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        attached, *_ = planner.open_suction(agent_id=0) 
        pose1[2] += 0.1
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        pose1[1] += 0.4
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)    
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.card, actor_data=env.annotation_data['card'], pre_dis=0, id=1)
        pose2[1] += 0.1
        planner.move_to_pose_with_screw(pose=pose2, move_id=1)
        pose2[1] -= 0.1
        pose2[2] -= 0.0015
        planner.move_to_pose_with_screw(pose=pose2, move_id=1)
        planner.close_gripper(close_id=1)
        planner.close_suction(agent_id=0)
        planner._hold_current_pose(steps=2, agent_id=1)
        pose2[1] += 0.25
        planner.move_to_pose_with_screw(pose=pose2, move_id=1)
        pose3 = planner.get_target_pose_w_labeled_direction(actor=env.terminal, actor_data=env.annotation_data['terminal'], pre_dis=0, id=0)
        pose3[2] += 0.25
        planner.move_to_pose_with_screw(pose=pose3, move_id=1)
        pose3[2] -= 0.25
        planner.move_to_pose_with_screw(pose=pose3, move_id=1)
        # print("Card Position:", env.card.pose.p)
        # print("Terminal Position:", env.terminal.pose.p)
        res = planner.open_gripper(open_id=1)

    # Gripper
    if tool_modes == ["gripper", "gripper"]:
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.card, actor_data=env.annotation_data['card'], pre_dis=0, id=2)
        pose1[1] -= 0.2
        pose1[2] += 0.2
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        pose1[2] -= 0.2
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.card, actor_data=env.annotation_data['card'], pre_dis=0, id=2)
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        planner.close_gripper(close_id=0)
        pose1[2] += 0.2
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        pose1[1] += 0.2
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.card, actor_data=env.annotation_data['card'], pre_dis=0, id=1)
        pose2[1] += 0.2
        planner.move_to_pose_with_screw(pose=pose2, move_id=1)
        pose2 = planner.get_grasp_pose_w_labeled_direction(actor=env.card, actor_data=env.annotation_data['card'], pre_dis=0, id=1)
        planner.move_to_pose_with_screw(pose=pose2, move_id=1)
        planner.close_gripper(close_id=1)
        planner.open_gripper(open_id=0)

        pose3 = planner.get_target_pose_w_labeled_direction(actor=env.terminal, actor_data=env.annotation_data['terminal'], pre_dis=0, id=0)
        pose3[2] += 0.25
        planner.move_to_pose_with_screw(pose=pose3, move_id=1)
        pose3[2] -= 0.25
        planner.move_to_pose_with_screw(pose=pose3, move_id=1)
        # # print("Card Position:", env.card.pose.p)
        # # print("Terminal Position:", env.terminal.pose.p)
        res = planner.open_gripper(open_id=1)

    planner.close()
    return res
