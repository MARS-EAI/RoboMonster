# solutions/cube_pick_place.py
import numpy as np
import sapien

from tasks import SuctionCardEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: SuctionCardEnv, seed=None, debug=False, vis=False, tool_modes=None):
    env.reset(seed=seed)
    if tool_modes is None:
        tool_modes = "gripper"
    base_pose = [env.agent.robot.pose]
    is_multi_agent = False
    
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
    
    # Gripper Actions:
    if tool_modes == 'gripper':
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.card, actor_data=env.annotation_data['card'], pre_dis=0, id=1)
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        planner.close_gripper(close_id=0)
        pose1[2] += 0.2
        res = planner.move_to_pose_with_screw(pose=pose1, move_id=0)


    # Suction Actions:
    else:
        tool = getattr(env.unwrapped, "_shared_suction_tool", None)
        assert tool is not None, "SuctionTool not found; make sure SuctionActionWrapper is applied."
        tool.clear_filters()
        tool.allow_only_names(contains=["card"])          
        tool.disallow_names(contains=["cube","table", "floor", "ground"])
        env = env.unwrapped
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.card, actor_data=env.annotation_data['card'], pre_dis=0, id=0)
        # pose1[2] += 0.038
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        attached, *_ = planner.open_suction(agent_id=0) 
        pose1[2] += 0.2
        res = planner.move_to_pose_with_screw(pose=pose1, move_id=0)

    planner.close()
    return res

