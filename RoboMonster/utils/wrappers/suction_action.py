# utils/wrappers/suction_action.py
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.suction.suction import SuctionTool, CircleTool
from typing import Optional, List


class SuctionActionWrapper(gym.Wrapper):
    """
    Generic single-/multi-agent wrapper to unify suction and gripper control.

    External action (per agent):
      - If agent is in `stick_agents`: action = [arm_qpos] (no tool slot, 7-dim for Panda)
      - Otherwise: action = [arm_qpos, tool] where:
          * agents in `suction_agents`: `tool` is suction on/off (>= on_threshold -> ON),
            and the underlying gripper channel is held at `gripper_hold_value`
          * agents in `circle_agents`: `tool` controls circle: <= -1 -> CLOSE, >= +1 -> OPEN
          * other agents: `tool` passes through to the underlying gripper channel

    Internal mapping to env:
      - pd_joint_pos     -> [qpos, gripper] or [qpos] if no gripper channel
      - pd_joint_pos_vel -> [qpos, zeros_like(qpos), gripper] or [qpos, zeros_like(qpos)]
    """

    def __init__(
        self,
        env,
        on_threshold: float = 0.0,
        gripper_hold_value: float = -1.0,
        probe_internal_steps: int = 0,
        debug: bool = False,
        suction_agents: Optional[List[str]] = None,
        stick_agents: Optional[List[str]] = None,
        circle_agents: Optional[List[str]] = None,
    ):
        super().__init__(env)
        self.on_threshold = float(on_threshold)
        self.gripper_hold_value = float(gripper_hold_value)
        self.probe_internal_steps = int(probe_internal_steps)
        self.debug = bool(debug)

        base = self.unwrapped
        cm = getattr(base, "control_mode", None)
        try:
            if hasattr(base, "agent") and hasattr(base.agent, "agents"):
                cm = {ag.uid: ag.control_mode for ag in base.agent.agents}
            elif hasattr(base, "agent"):
                cm = base.agent.control_mode
        except Exception:
            pass
        self._raw_control_mode = cm

        from gymnasium import spaces
        self._is_multi = isinstance(env.action_space, spaces.Dict)
        if self._is_multi:
            self.agent_keys = list(env.action_space.spaces.keys())
        else:
            self.agent_keys = ["__single__"]

        if self._is_multi:
            if not isinstance(self._raw_control_mode, dict):
                self._control_mode = {k: self._raw_control_mode for k in self.agent_keys}
            else:
                self._control_mode = {k: self._raw_control_mode.get(k) for k in self.agent_keys}
        else:
            if isinstance(self._raw_control_mode, dict):
                first_key = next(iter(self._raw_control_mode.keys()))
                self._control_mode = self._raw_control_mode[first_key]
            else:
                self._control_mode = self._raw_control_mode

        self._suction_agents = set((suction_agents or []))
        self._stick_agents   = set((stick_agents or []))
        self._circle_agents  = set((circle_agents or []))

        missing_s = self._suction_agents.difference(set(self.agent_keys))
        missing_n = self._stick_agents.difference(set(self.agent_keys))
        missing_c = self._circle_agents.difference(set(self.agent_keys))
        if self.debug and (missing_s or missing_n or missing_c):
            if missing_s:
                print(f"[suction] WARNING: suction_agents {missing_s} not in env agent_keys {self.agent_keys}")
            if missing_n:
                print(f"[suction] WARNING: stick_agents {missing_n} not in env agent_keys {self.agent_keys}")
            if missing_c:
                print(f"[circle] WARNING: circle_agents {missing_c} not in env agent_keys {self.agent_keys}")

        self._per_agent_cfg = {}
        if self._is_multi:
            new_spaces = {}
            for k, sp in env.action_space.spaces.items():
                sub_mode = self._control_mode.get(k) if isinstance(self._control_mode, dict) else self._control_mode
                is_circle = (k in self._circle_agents)
                is_stick  = (k in self._stick_agents)
                has_gripper_channel = (not is_stick) and (not is_circle)
                has_tool_slot_ext = (k in self._suction_agents) or is_circle or ((k not in self._stick_agents) and (k not in self._circle_agents)) or is_stick
                dofs = _infer_arm_dofs(sp, sub_mode, assume_gripper=has_gripper_channel)
                out_dim = dofs + (1 if has_tool_slot_ext else 0)
                out_sp = spaces.Box(low=-np.inf, high=np.inf, shape=(out_dim,), dtype=np.float32)

                self._per_agent_cfg[k] = dict(
                    dofs=dofs,
                    mode=sub_mode,
                    has_gripper_channel=has_gripper_channel,
                    has_tool_slot_ext=has_tool_slot_ext,
                    is_circle=is_circle,
                )
                new_spaces[k] = out_sp
            self.action_space = spaces.Dict(new_spaces)
        else:
            mode = self._control_mode if (isinstance(self._control_mode, str) or self._control_mode is None) else None
            is_circle = ("__single__" in self._circle_agents)
            is_stick  = ("__single__" in self._stick_agents)
            has_gripper_channel = (not is_stick) and (not is_circle)
            has_tool_slot_ext = ("__single__" in self._suction_agents) or is_circle or (("__single__" not in self._stick_agents) and ("__single__" not in self._circle_agents)) or is_stick
            dofs = _infer_arm_dofs(env.action_space, mode, assume_gripper=has_gripper_channel)
            out_dim = dofs + (1 if has_tool_slot_ext else 0)
            out_sp = spaces.Box(low=-np.inf, high=np.inf, shape=(out_dim,), dtype=np.float32)
            self._per_agent_cfg["__single__"] = dict(
                dofs=dofs, mode=mode,
                has_gripper_channel=has_gripper_channel,
                has_tool_slot_ext=has_tool_slot_ext,
                is_circle=is_circle,
            )
            self.action_space = out_sp

        self._suction = SuctionTool(env=base, debug=debug)
        self._fingers_ready = False
        self._warned_no_fingers = False
        setattr(base, "_shared_suction_tool", self._suction)
        self._refresh_fingers_if_needed(force=False)

        self._circle = CircleTool(env=base, debug=debug)
        self._anchor_ready = False
        setattr(base, "_shared_circle_tool", self._circle)
        self._refresh_circle_if_needed(force=False)

        if self.debug:
            print("[suction] is_multi      =", self._is_multi)
            print("[suction] agent_keys    =", self.agent_keys)
            print("[suction] control_mode  =", self._control_mode)
            dbg = {k: dict(dofs=v["dofs"], ext_tool=v["has_tool_slot_ext"], grip_ch=v["has_gripper_channel"], circle=v["is_circle"]) for k, v in self._per_agent_cfg.items()}
            print("[suction] per_agent_cfg =", dbg)
            print("[suction] suction_agents=", self._suction_agents)
            print("[suction] stick_agents  =", self._stick_agents)
            print("[circle]  circle_agents =", self._circle_agents)

    # ======= Helper: collect circle anchor/probe/hand links =======
    def _refresh_circle_if_needed(self, force: bool = False):
        """
        Locate and set the circle tool anchor (panda_sleeve), optional probe (panda_hand_tcp),
        and panda_hand link on the proper agent. Runs once, unless force=True.
        """
        if self._anchor_ready and not force:
            return
        base = self.unwrapped
        parent = None  # panda_sleeve
        probe  = None  # panda_hand_tcp
        hand   = None  # panda_hand

        def _base(name: str) -> str:
            if not isinstance(name, str):
                return ""
            return name.split("-", 1)[0]

        def _uid_matches_circle(uid: str) -> bool:
            return (uid in self._circle_agents) or (_base(uid) in self._circle_agents)

        try:
            agents = []
            if not self._is_multi:
                if hasattr(base, "agent"):
                    agents = [base.agent]
            else:
                if hasattr(base, "agent") and hasattr(base.agent, "agents"):
                    agents = list(base.agent.agents)

            # Prefer agents explicitly listed in circle_agents; otherwise, fall back to all agents
            picked = []
            if self._circle_agents:
                for ag in agents:
                    uid = getattr(ag, "uid", getattr(ag, "robot_uid", ""))
                    if _uid_matches_circle(uid):
                        picked.append(ag)
            if not picked:
                picked = agents

            # Scan for required link names on the chosen agent
            for ag in picked:
                links = []
                try:
                    links = ag.robot.get_links()
                except Exception:
                    pass

                this_parent = None
                this_probe  = None
                this_hand   = None
                for l in links or []:
                    nm = l.get_name() if hasattr(l, "get_name") else getattr(l, "name", "")
                    if nm == "panda_sleeve":
                        this_parent = l
                    elif nm == "panda_hand_tcp":
                        this_probe = l
                    elif nm == "panda_hand":
                        this_hand = l

                if this_parent is not None:
                    parent, probe, hand = this_parent, this_probe, this_hand
                    break
        except Exception:
            pass

        if parent:
            self._circle.set_anchor_links(parent, probe, hand)
            self._anchor_ready = True
            if self.debug:
                pn = parent.get_name() if hasattr(parent, "get_name") else "?"
                hb = hand.get_name() if (hand and hasattr(hand, "get_name")) else None
                pr = probe.get_name() if (probe and hasattr(probe, "get_name")) else None
                print(f"[circle] anchor set: parent={pn} probe={pr} hand={hb}")
        else:
            self._anchor_ready = False
            # if self.debug:
            #     print("[circle] anchor not ready yet (need links: panda_sleeve; probe optional)")

    def _refresh_fingers_if_needed(self, force: bool = False):
        """
        Collect suction finger/pad links once (unless force=True), for agents in suction_agents.
        """
        if self._fingers_ready and not force:
            return
        base = self.unwrapped
        finger_links = []
        try:
            if not self._is_multi:
                if hasattr(base, "agent") and hasattr(base.agent, "robot"):
                    links = base.agent.robot.get_links()
                    finger_links.extend([l for l in links if _is_finger(l)])
            else:
                if hasattr(base, "agent") and hasattr(base.agent, "agents"):
                    for ag in base.agent.agents:
                        if ag.uid in self._suction_agents:
                            links = ag.robot.get_links()
                            finger_links.extend([l for l in links if _is_finger(l)])
        except Exception:
            pass

        if len(finger_links) > 0:
            self._suction.set_fingers(finger_links)
            self._fingers_ready = True
            self._warned_no_fingers = False
        else:
            # Warn only once to avoid spam
            if self.debug and not self._warned_no_fingers:
                print("[suction] fingers = [] (will try again after reset / on first step)")
                self._warned_no_fingers = True

    # ======= Basic utilities =======
    def _want_on(self, flag: float) -> bool:
        """Interpret tool flag by threshold (>= threshold means ON)."""
        return float(flag) >= self.on_threshold

    def _apply_flag_side_effects(self, want_on: bool):
        """
        Trigger a one-shot suction grab/release; can probe a few internal steps for stability.
        """
        attached = self._suction.is_attached()
        if want_on and not attached:
            ok = self._suction.grab_once()
            if (not ok) and self.probe_internal_steps > 0:
                for _ in range(self.probe_internal_steps):
                    self._suction.update_attachment()
        elif (not want_on) and attached:
            self._suction.release()

    @staticmethod
    def _map_to_inner(qpos: np.ndarray, mode: str, gripper_val: Optional[float], has_gripper_channel: bool) -> np.ndarray:
        """
        Map outer action (arm_qpos + optional gripper value) to inner env action
        according to control mode and presence of gripper channel.
        """
        if (mode is None) or (mode == "pd_joint_pos"):
            if has_gripper_channel:
                return np.hstack([qpos, float(gripper_val)]).astype(np.float32)
            return np.asarray(qpos, dtype=np.float32)
        elif mode == "pd_joint_pos_vel":
            zeros = np.zeros_like(qpos, dtype=np.float32)
            if has_gripper_channel:
                return np.hstack([qpos, zeros, float(gripper_val)]).astype(np.float32)
            return np.hstack([qpos, zeros]).astype(np.float32)
        else:
            raise NotImplementedError(
                f"SuctionActionWrapper only supports pd_joint_pos / pd_joint_pos_vel for now; control_mode={mode}"
            )

    # ======= Gym API =======
    def step(self, action):
        if not self._fingers_ready:
            self._refresh_fingers_if_needed(force=False)
        if not self._anchor_ready:
            self._refresh_circle_if_needed(force=False)

        self._suction.update_attachment()
        self._circle.update_attachment()

        if self._is_multi:
            inner_action = {}
            for k, a in action.items():
                a = np.asarray(a).reshape(-1)
                cfg = self._per_agent_cfg[k]
                dofs, mode = cfg["dofs"], cfg["mode"]
                has_tool_ext = cfg["has_tool_slot_ext"]
                has_grip_ch  = cfg["has_gripper_channel"]
                is_circle    = cfg["is_circle"]

                expected = dofs + (1 if has_tool_ext else 0)
                if a.size != expected:
                    if (k in self._stick_agents) and has_tool_ext and (a.size == dofs):
                        a = np.hstack([a, 0.0]).astype(np.float32)
                    elif (not has_tool_ext) and (a.size == dofs + 1):
                        if self.debug:
                            print(f"[suction] WARN: agent '{k}' expects {dofs}-dim (stick), got {a.size}; ignoring last dim once.")
                        a = a[:dofs]
                    else:
                        raise ValueError(f"Action dim mismatch for agent '{k}': expect {expected}, got {a.size}")

                if has_tool_ext:
                    qpos, tool = a[:dofs], a[-1]
                else:
                    qpos, tool = a[:dofs], None

                if k in self._suction_agents:
                    self._apply_flag_side_effects(self._want_on(tool if tool is not None else -1.0))
                    gripper_val = self.gripper_hold_value if has_grip_ch else None
                elif is_circle:
                    if tool is not None:
                        if float(tool) <= -1.0:
                            self._circle.grab_once()
                        elif float(tool) >= +1.0:
                            self._circle.release()
                    gripper_val = None
                else:
                    gripper_val = float(tool) if (tool is not None and has_grip_ch) else None

                inner_action[k] = self._map_to_inner(qpos, mode, gripper_val, has_grip_ch)
        else:
            a = np.asarray(action).reshape(-1)
            cfg = self._per_agent_cfg["__single__"]
            dofs, mode = cfg["dofs"], cfg["mode"]
            has_tool_ext = cfg["has_tool_slot_ext"]
            has_grip_ch  = cfg["has_gripper_channel"]
            is_circle    = cfg["is_circle"]

            expected = dofs + (1 if has_tool_ext else 0)
            if a.size != expected:
                if ("__single__" in self._stick_agents) and has_tool_ext and (a.size == dofs):
                    a = np.hstack([a, 0.0]).astype(np.float32)
                elif (not has_tool_ext) and (a.size == dofs + 1):
                    if self.debug:
                        print(f"[suction] WARN: single agent expects {dofs}-dim (stick), got {a.size}; ignoring last dim once.")
                    a = a[:dofs]
                else:
                    raise ValueError(f"Action dim mismatch for single agent: expect {expected}, got {a.size}")

            if has_tool_ext:
                qpos, tool = a[:dofs], a[-1]
            else:
                qpos, tool = a[:dofs], None

            if "__single__" in self._suction_agents:
                self._apply_flag_side_effects(self._want_on(tool if tool is not None else -1.0))
                gripper_val = self.gripper_hold_value if has_grip_ch else None
            elif is_circle:
                if tool is not None:
                    if float(tool) <= -1.0:
                        self._circle.grab_once()
                    elif float(tool) >= +1.0:
                        self._circle.release()
                gripper_val = None
            else:
                gripper_val = float(tool) if (tool is not None and has_grip_ch) else None

            inner_action = self._map_to_inner(qpos, mode, gripper_val, has_grip_ch)

        obs, rew, term, trunc, info = self.env.step(inner_action)

        self._suction.update_attachment()
        self._circle.update_attachment()
        attached_suction = bool(self._suction.is_attached())
        attached_circle  = bool(self._circle.is_attached())

        if self._is_multi and isinstance(info, dict) \
                and set(self.agent_keys).issubset(set(info.keys())) \
                and all(isinstance(info[k], dict) for k in self.agent_keys):
            out = dict(info)
            for k in self.agent_keys:
                sub = dict(out[k])
                sub["suction_attached"] = (attached_suction if k in self._suction_agents else False)
                sub["circle_attached"]  = (attached_circle  if k in self._circle_agents  else False)
                out[k] = sub
            info = out
        else:
            base = dict(info) if isinstance(info, dict) else {}
            base["suction_attached_map"] = {k: (attached_suction if k in self._suction_agents else False) for k in self.agent_keys}
            base["circle_attached_map"]  = {k: (attached_circle  if k in self._circle_agents  else False) for k in self.agent_keys}
            if not self._is_multi:
                base["suction_attached"] = attached_suction
                base["circle_attached"]  = attached_circle
            info = base

        return obs, rew, term, trunc, info

    def reset(self, **kwargs):
        """
        Reset wrapper state and tools, then pass through to env.reset(...).
        """
        # Clear attachments and readiness flags
        self._suction.release()
        self._fingers_ready = False
        self._warned_no_fingers = False

        self._circle.release()
        self._anchor_ready = False

        # Reset env and re-collect links for tools
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info = ret
        else:
            obs, info = ret, {}

        self._refresh_fingers_if_needed(force=True)
        self._refresh_circle_if_needed(force=True)
        return obs, info


# ======= Helper functions =======
def _is_finger(link) -> bool:
    """
    Return True if a link looks like a finger/pad (name contains 'finger' or 'pad').
    """
    name = link.get_name() if hasattr(link, "get_name") else getattr(link, "name", "")
    low = name.lower()
    return ("finger" in low) or ("pad" in low)


def _infer_arm_dofs(action_space: spaces.Space, control_mode: str, assume_gripper: bool) -> int:
    """
    Infer arm DOFs from low-level Box action space and control_mode.
    If assume_gripper=True, subtract a gripper channel (or two channels for pos+vel).
    """
    if not isinstance(action_space, spaces.Box):
        raise TypeError(f"Only Box action spaces are supported, got {type(action_space)}")
    dim = int(np.prod(action_space.shape))
    if dim <= 0:
        raise ValueError(f"Invalid action dimension: {action_space.shape}")

    g = 1 if assume_gripper else 0
    if control_mode == "pd_joint_pos":
        return dim - g
    elif control_mode == "pd_joint_pos_vel":
        return (dim - g) // 2
    else:
        # Fallback if a rare mode is used; keep consistent with prior logic
        return dim - g
