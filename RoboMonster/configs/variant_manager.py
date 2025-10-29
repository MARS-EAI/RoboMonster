from copy import deepcopy

def _lock_gripper_cfg(agent_pos):
    """
    Suction-style tool:
    - Keep 9-DOF Panda model (7 arm joints + 2 finger joints)
    - Force the last 2 finger joints closed (0.0)
    - Zero out their randq_scale so reset() won't re-randomize them open
    """
    q = list(agent_pos["qpos"])
    rq = list(agent_pos["randq_scale"])

    if len(q) >= 9:
        q[-2] = 0.0
        q[-1] = 0.0
    if len(rq) >= 9:
        rq[-2] = 0.0
        rq[-1] = 0.0

    agent_pos["qpos"] = q
    agent_pos["randq_scale"] = rq
    return agent_pos


def _drop_gripper_cfg(agent_pos):
    """
    Stick / circle tool:
    - Treat arm as pure 7-DOF tool arm (no finger joints at all)
    - Truncate both qpos and randq_scale to first 7 joints
    """
    q = list(agent_pos["qpos"])[:7]
    rq = list(agent_pos["randq_scale"])[:7]

    agent_pos["qpos"] = q
    agent_pos["randq_scale"] = rq
    return agent_pos


def apply_variant_to_yaml(cfg_dict, task_name, variant):
    """
    Mutate qpos / randq_scale inside cfg["agents"] based on:
      - task_name       (snake_case, e.g. 'swipe_card', 'place_badminton')
      - variant         ('gripper' baseline vs 'ours')

    NOTE: we do NOT touch randp_scale (base xyz noise); it is always length 3.
    """
    cfg = deepcopy(cfg_dict)
    agents = cfg["agents"]

    # ----- single-agent tasks -----
    if task_name in ["suction_card", "circle_vase", "pick_pokeball"]:
        assert len(agents) == 1, "single-agent task should start with 1 agent"
        ag = agents[0]

        if variant == "gripper":
            # vanilla gripper, do nothing
            pass

        elif variant == "ours":
            if task_name == "suction_card":
                # same Panda arm, but lock gripper as suction
                ag["pos"] = _lock_gripper_cfg(ag["pos"])

            elif task_name in ["circle_vase", "pick_pokeball"]:
                # circle tool arm => 7DOF no finger
                ag["pos"] = _drop_gripper_cfg(ag["pos"])

            else:
                raise RuntimeError("unknown single-agent ours variant")

        else:
            raise RuntimeError("variant must be 'gripper' or 'ours'")

        cfg["agents"] = [ag]

    # ----- dual-agent tasks -----
    elif task_name in ["swipe_card", "place_badminton"]:
        assert len(agents) == 2, "dual-agent task should start with 2 agents"
        ag0, ag1 = agents

        if variant == "gripper":
            # both are normal grippers
            pass

        elif variant == "ours":
            if task_name == "swipe_card":
                # panda-0 becomes suction (lock gripper closed),
                # panda-1 stays normal gripper
                ag0["pos"] = _lock_gripper_cfg(ag0["pos"])

            elif task_name == "place_badminton":
                # panda-0 stays normal gripper,
                # panda_stick-1 becomes 7DOF stick tool (drop fingers)
                ag1["pos"] = _drop_gripper_cfg(ag1["pos"])

            else:
                raise RuntimeError("unknown dual-agent ours variant")

        else:
            raise RuntimeError("variant must be 'gripper' or 'ours'")

        cfg["agents"] = [ag0, ag1]

    else:
        raise RuntimeError(f"unknown task_name {task_name}")

    return cfg


def get_tool_modes(task_name, variant):
    # tool_modes feeds the motion planner:
    #   e.g. ["suction","gripper"] means arm0 uses suction logic, arm1 uses gripper logic
    if task_name in ["suction_card", "circle_vase", "pick_pokeball"]:
        if variant == "gripper":
            return ["gripper"]
        elif variant == "ours":
            if task_name == "suction_card":
                return ["suction"]
            elif task_name in ["circle_vase", "pick_pokeball"]:
                return ["circle"]

    elif task_name == "swipe_card":
        if variant == "gripper":
            return ["gripper", "gripper"]
        elif variant == "ours":
            # panda-0 suction, panda-1 normal gripper
            return ["suction", "gripper"]

    elif task_name == "place_badminton":
        if variant == "gripper":
            return ["gripper", "gripper"]
        elif variant == "ours":
            # panda-0 gripper, panda_stick-1 stick tool
            return ["gripper", "stick"]

    raise RuntimeError(f"no tool_modes for {task_name} / {variant}")


def get_wrapper_kwargs(task_name, variant):
    # These kwargs go into SuctionActionWrapper.
    # We declare which agent names get suction/stick/circle semantics.
    base = dict(
        gripper_hold_value=-1.0,
        probe_internal_steps=3,
        debug=True,
        suction_agents=[],
        stick_agents=[],
        circle_agents=[],
    )

    if variant == "gripper":
        return base

    # ours:
    if task_name == "suction_card":
        base["suction_agents"] = ["__single__"]

    elif task_name == "circle_vase":
        base["circle_agents"] = ["__single__"]

    elif task_name == "pick_pokeball":
        base["circle_agents"] = ["__single__"]

    elif task_name == "swipe_card":
        # panda-0 is suction
        base["suction_agents"] = ["panda-0"]

    elif task_name == "place_badminton":
        # second arm is the stick tool
        base["stick_agents"] = ["panda_stick-1"]

    else:
        raise RuntimeError(f"no wrapper cfg for {task_name}/{variant}")

    return base


def get_robot_uids(task_name, variant):
    # Which robot class to actually spawn for each arm.
    # IMPORTANT:
    #  - suction_card ours: still ("panda",) because it's the same Panda body, just locked gripper
    #  - circle_vase / pick_pokeball ours: spawn ("panda_circle",) = 7DOF circle tool
    #  - place_badminton ours: ("panda","panda_stick")
    #
    if task_name in ["suction_card", "circle_vase", "pick_pokeball"]:
        if variant == "gripper":
            return ("panda",)
        elif variant == "ours":
            if task_name == "suction_card":
                return ("panda",)
            elif task_name in ["circle_vase", "pick_pokeball"]:
                return ("panda_circle",)

    elif task_name == "swipe_card":
        if variant == "gripper":
            return ("panda", "panda")
        elif variant == "ours":
            # suction on arm0 but it's still a Panda,
            # arm1 is still a normal Panda with gripper
            return ("panda", "panda")

    elif task_name == "place_badminton":
        if variant == "gripper":
            return ("panda", "panda")
        elif variant == "ours":
            # first is standard Panda gripper,
            # second is Panda-with-stick-tool
            return ("panda", "panda_stick")

    raise RuntimeError(f"no robot_uids for {task_name}/{variant}")
