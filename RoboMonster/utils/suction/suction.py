# utils/suction/suction.py
from __future__ import annotations

import numpy as np
import sapien
import sapien.physx as physx
from sapien import Pose
from typing import Optional, Sequence, List, Iterable, Callable, Set


class SuctionTool:
    """
    Stable "kinematic adhesion" implementation + target filtering:
      - Contact detection: iterate scene.get_contacts() to find contacts between fingers and targets
      - Attach: set the target set_kinematic(True) and record the finger target relative pose
      - Follow: each step update_attachment() forcibly sets the target pose
      - Release: restore to a dynamic body and zero its linear/angular velocities
      - Target filtering: allow/deny lists (by name/entity), custom filter function, optional mass range

    Usage examples (choose any or combine):
        tool.allow_only_names(contains=["cube", "card"])
        tool.disallow_names(contains=["table", "floor", "ground"])
        tool.allow_only_entities([env.cube])
        tool.set_filter_fn(lambda e: "cube" in tool._pretty_name(e))
        tool.set_mass_range(min_mass=0.01, max_mass=2.0)
    """

    def __init__(self, env, debug: bool = True):
        self.scene: sapien.Scene = env.scene
        self.debug = bool(debug)

        # fingers
        self._finger_links: List[sapien.Entity] = []

        # attachment state
        self._attached_comp: Optional[physx.PhysxRigidDynamicComponent] = None
        self._parent_link_comp: Optional[physx.PhysxArticulationLinkComponent] = None
        self._relative_pose: Optional[Pose] = None

        # prevent repeated warning spam
        self._warned_no_fingers_once = False

        # Target filtering: allow/deny lists + custom predicate + mass thresholds (optional)
        self._allow_exact: Set[str] = set()
        self._allow_contains: Set[str] = set()
        self._deny_exact: Set[str] = set()
        self._deny_contains: Set[str] = set()
        self._allow_entities: Set[sapien.Entity] = set()
        self._filter_fn: Optional[Callable[[sapien.Entity], bool]] = None
        self._min_mass: Optional[float] = None
        self._max_mass: Optional[float] = None

    # ======== Public API ========
    def set_fingers(self, links: Sequence):
        """Specify the finger/pad links treated as the suction surface."""
        self._finger_links = list(links)
        self._warned_no_fingers_once = False
        if self.debug:
            def nm(x): return x.get_name() if hasattr(x, "get_name") else getattr(x, "name", "?")
            print("[suction] fingers =", [nm(l) for l in self._finger_links])

    def has_fingers(self) -> bool:
        return len(self._finger_links) > 0

    def is_attached(self) -> bool:
        return self._attached_comp is not None

    def release(self):
        """
        Release the object and restore its physics to a dynamic body.
        """
        if not self.is_attached():
            return
        try:
            self._attached_comp.set_kinematic(False)
            # Zero velocities to avoid explosions at release
            self._attached_comp.linear_velocity = np.zeros(3)
            self._attached_comp.angular_velocity = np.zeros(3)
            if self.debug:
                print(f"[suction] Released '{self._pretty_name(self._attached_comp.entity)}' and re-enabled physics.")
        finally:
            self._attached_comp = None
            self._parent_link_comp = None
            self._relative_pose = None

    def grab_once(self) -> bool:
        """
        Try a one-shot "grab" based on contacts in the current frame:
          - If any finger contacts a dynamic rigid body that is NOT a robot articulation,
            and it passes target filters, set it kinematic and save the relative pose.
        """
        if self.is_attached():
            return True

        if not self.has_fingers():
            if self.debug and not self._warned_no_fingers_once:
                print("[suction] WARNING: No finger links set. Call set_fingers([...]) first.")
                self._warned_no_fingers_once = True
            return False

        finger_names = {self._pretty_name(link) for link in self._finger_links}

        for c in self.scene.get_contacts():
            side0 = self._collect_side(c, 0)
            side1 = self._collect_side(c, 1)

            for s_touch, s_other in ((side0, side1), (side1, side0)):
                finger_entity = self._pick_by_name(s_touch, finger_names)
                if not finger_entity:
                    continue

                target_entity = self._pick_valid_target(s_other)
                if not target_entity:
                    continue

                finger_comp = finger_entity.find_component_by_type(physx.PhysxArticulationLinkComponent)
                target_comp = target_entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
                if not (finger_comp and target_comp):
                    continue

                # Attach
                target_comp.set_kinematic(True)
                self._attached_comp = target_comp
                self._parent_link_comp = finger_comp
                self._relative_pose = finger_comp.pose.inv() * target_comp.pose

                if self.debug:
                    print(f"[suction] Contact! Attach '{self._pretty_name(target_entity)}' to '{self._pretty_name(finger_entity)}' (kinematic).")
                return True

        return False

    def update_attachment(self):
        """
        If attached, update the target object's world pose according to the saved relative pose.
        """
        if not self.is_attached():
            return
        new_pose = self._parent_link_comp.pose * self._relative_pose
        self._attached_comp.entity.set_pose(new_pose)

    # ======== Target filter configuration API ========
    def allow_only_names(self, exact: Iterable[str] = None, contains: Iterable[str] = None):
        """Allow only these names (exact for full match / contains for substring match)."""
        if exact:
            self._allow_exact.update(map(str, exact))
        if contains:
            self._allow_contains.update(map(str, contains))

    def disallow_names(self, exact: Iterable[str] = None, contains: Iterable[str] = None):
        """
        Disallow these names (exact for full match / contains for substring match).
        """
        if exact:
            self._deny_exact.update(map(str, exact))
        if contains:
            self._deny_contains.update(map(str, contains))

    def allow_only_entities(self, entities: Iterable[sapien.Entity]):
        """
        Allow only these specific entities (strongest constraint).
        """
        self._allow_entities.update(entities)

    def set_mass_range(self, min_mass: float = None, max_mass: float = None):
        """
        Optional: filter by mass range (ignored if the component doesn't expose mass).
        """
        self._min_mass = min_mass
        self._max_mass = max_mass

    def set_filter_fn(self, fn: Optional[Callable[[sapien.Entity], bool]]):
        """
        Attach a custom filter: fn(entity) -> bool. Returns True to allow attaching.
        """
        self._filter_fn = fn

    def clear_filters(self):
        """
        Clear all filtering conditions.
        """
        self._allow_exact.clear()
        self._allow_contains.clear()
        self._deny_exact.clear()
        self._deny_contains.clear()
        self._allow_entities.clear()
        self._filter_fn = None
        self._min_mass = None
        self._max_mass = None

    # ======== Utilities ========
    def _collect_side(self, contact, idx: int):
        out = []
        try:
            # SAPIEN 2.x: contact.bodies[idx].entity
            if hasattr(contact, "bodies"):
                b = contact.bodies[idx]
                ent = getattr(b, "entity", None)
                if ent is not None:
                    out.append(ent)
            else:
                # Compatibility with legacy fields (actor0/actor1)
                if idx == 0 and hasattr(contact, "actor0"):
                    ent = getattr(contact.actor0, "entity", None)
                    if ent is not None:
                        out.append(ent)
                if idx == 1 and hasattr(contact, "actor1"):
                    ent = getattr(contact.actor1, "entity", None)
                    if ent is not None:
                        out.append(ent)
        except Exception:
            pass
        return out

    def _pick_by_name(self, entities, names: set[str]):
        for e in entities:
            if self._pretty_name(e) in names:
                return e
        return None

    def _pick_valid_target(self, entities):
        """
        Select only dynamic rigid-body entities that are NOT robot articulations, then apply filters:
          - Allow lists (entities/names exact/contains)
          - Deny lists
          - Custom filter function
          - Mass range (if retrievable)
        """
        for e in entities:
            if e is None:
                continue

            # Exclude robot links (they have an articulation link component)
            if e.find_component_by_type(physx.PhysxArticulationLinkComponent):
                continue

            comp = e.find_component_by_type(physx.PhysxRigidDynamicComponent)
            if not comp:
                continue  # Only pick dynamic bodies

            name = self._pretty_name(e)

            # ======== Allow/Deny lists & custom filter ========
            if self._allow_entities and e not in self._allow_entities:
                continue
            if self._allow_exact and name not in self._allow_exact:
                continue
            if self._allow_contains and not any(k in name for k in self._allow_contains):
                continue
            if self._deny_exact and name in self._deny_exact:
                continue
            if self._deny_contains and any(k in name for k in self._deny_contains):
                continue

            # Mass filtering (best-effort; ignore if we can't retrieve it)
            if (self._min_mass is not None) or (self._max_mass is not None):
                m = None
                try:
                    # APIs may differ by version; try both and skip mass filtering if retrieval fails
                    if hasattr(comp, "get_mass"):
                        m = float(comp.get_mass())
                    elif hasattr(comp, "mass"):
                        m = float(comp.mass)
                except Exception:
                    m = None
                if m is not None:
                    if (self._min_mass is not None) and (m < self._min_mass):
                        continue
                    if (self._max_mass is not None) and (m > self._max_mass):
                        continue

            if self._filter_fn and (not self._filter_fn(e)):
                continue

            return e
        return None

    def _pretty_name(self, obj) -> str:
        if obj is None:
            return "None"
        if hasattr(obj, "get_name"):
            return obj.get_name()
        return getattr(obj, "name", type(obj).__name__)
    
# ===== CircleTool ========

class CircleTool(SuctionTool):
    """
    use the same logic with SuctionTool
    """
    def __init__(self, env, debug: bool = True):
        super().__init__(env, debug=debug)
        self._anchor_link_comp: Optional[physx.PhysxArticulationLinkComponent] = None  # panda_sleeve
        self._probe_link_comp: Optional[physx.PhysxArticulationLinkComponent] = None   # panda_hand_tcp
        self._hand_link_comp: Optional[physx.PhysxArticulationLinkComponent] = None    # panda_hand
        self._warned_no_anchor_once = False

    def set_anchor_links(self, parent_link, probe_link=None, hand_link=None):
        """
        - parent_link: expect panda_sleeve
        - probe_link : expect panda_hand_tcp
        - hand_link  : expect panda_hand
        """
        def _as_alc(x):
            if isinstance(x, physx.PhysxArticulationLinkComponent):
                return x
            if hasattr(x, "find_component_by_type"):
                return x.find_component_by_type(physx.PhysxArticulationLinkComponent)
            return None

        self._anchor_link_comp = _as_alc(parent_link)
        self._probe_link_comp  = _as_alc(probe_link)
        self._hand_link_comp   = _as_alc(hand_link)

        if self.debug:
            p = self._anchor_link_comp.entity if self._anchor_link_comp else None
            q = self._probe_link_comp.entity  if self._probe_link_comp  else None
            h = self._hand_link_comp.entity   if self._hand_link_comp   else None
            print(f"[circle] anchor(parent)={self._pretty_name(p) if p else None}, "
                  f"probe={self._pretty_name(q) if q else None}, "
                  f"hand={self._pretty_name(h) if h else None}")

    def is_ready(self) -> bool:
        return self._anchor_link_comp is not None

    def release(self):
        if not self.is_attached():
            return
        try:
            self._attached_comp.set_kinematic(False)
            self._attached_comp.linear_velocity = np.zeros(3)
            self._attached_comp.angular_velocity = np.zeros(3)
            if self.debug:
                print(f"[circle] Released '{self._pretty_name(self._attached_comp.entity)}'; physics re-enabled.")
        finally:
            self._attached_comp = None
            self._parent_link_comp = None
            self._relative_pose = None

    def update_attachment(self):
        if not self.is_attached():
            return
        parent = self._parent_link_comp or self._anchor_link_comp
        if parent is None:
            return
        new_pose = parent.pose * self._relative_pose
        self._attached_comp.entity.set_pose(new_pose)

    def grab_once(self) -> bool:
        if self.is_attached():
            return True
        finger_entities: Set[sapien.Entity] = set()
        if self._anchor_link_comp and self._anchor_link_comp.entity:
            finger_entities.add(self._anchor_link_comp.entity)
        if self._hand_link_comp and self._hand_link_comp.entity:
            finger_entities.add(self._hand_link_comp.entity)
        finger_name_whitelist = {"panda_sleeve", "panda_hand"}

        for c in self.scene.get_contacts():
            side0 = self._collect_side(c, 0)
            side1 = self._collect_side(c, 1)

            for s_touch, s_other in ((side0, side1), (side1, side0)):
                finger_entity = None
                for e in s_touch:
                    if e in finger_entities:
                        finger_entity = e
                        break
                if finger_entity is None:
                    for e in s_touch:
                        nm = self._pretty_name(e)
                        if nm in finger_name_whitelist:
                            finger_entity = e
                            break
                if finger_entity is None:
                    continue

                target_entity = self._pick_valid_target(s_other)
                if not target_entity:
                    continue

                finger_comp = finger_entity.find_component_by_type(physx.PhysxArticulationLinkComponent)
                target_comp = target_entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
                if not (finger_comp and target_comp):
                    continue

                target_comp.set_kinematic(True)
                self._attached_comp = target_comp
                self._parent_link_comp = finger_comp
                self._relative_pose = finger_comp.pose.inv() * target_comp.pose

                if self.debug:
                    print(f"[circle] ATTACH '{self._pretty_name(target_entity)}' "
                          f"to '{self._pretty_name(finger_entity)}' (kinematic).")
                return True

        # if self.debug:
            # print("[circle] no eligible contact on panda_hand/panda_sleeve; not attached.")
        return False

