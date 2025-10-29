# solutions/cube_pick_place.py
import numpy as np
import sapien

from tasks import PickPokeballEnv
from planner.motionplanner import PandaArmMotionPlanningSolver

def solve(env: PickPokeballEnv, seed=None, debug=False, vis=False, tool_modes=None):
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
    env = env.unwrapped

    if tool_modes == 'circle':
        tool = getattr(env.unwrapped, "_shared_circle_tool", None)
        assert tool is not None, "CircleTool not found; make sure SuctionActionWrapper is applied."
        tool.clear_filters()
        tool.allow_only_names(contains=["cube", "card", "vase", "pokeball"])       
        tool.disallow_names(contains=["table", "floor", "ground"])

        planner.open_circle(agent_id=0)
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.pokeball, actor_data=env.annotation_data['pokeball'], pre_dis=0, id=0)
        pose1[0] += 0.02
        pose1[2] += 0.2
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        pose1[2] -= 0.2
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        planner.close_circle(agent_id=0)
        pose1[2] += 0.3
        res = planner.move_to_pose_with_screw(pose=pose1, move_id=0)

    else:
        pose1 = planner.get_grasp_pose_w_labeled_direction(actor=env.pokeball, actor_data=env.annotation_data['pokeball'], pre_dis=0, id=0)
        pose1[0] += 0.02
        pose1[2] += 0.25
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        pose1[2] -= 0.25
        planner.move_to_pose_with_screw(pose=pose1, move_id=0)
        planner.close_gripper(close_id=0)
        pose1[2] += 0.3
        res = planner.move_to_pose_with_screw(pose=pose1, move_id=0)

    planner.close()
    return res
