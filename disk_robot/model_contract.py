from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINT_NAMES = (
    "leg_front_r_1",
    "leg_front_r_2",
    "leg_front_r_3",
    "leg_front_l_1",
    "leg_front_l_2",
    "leg_front_l_3",
    "leg_back_r_1",
    "leg_back_r_2",
    "leg_back_r_3",
    "leg_back_l_1",
    "leg_back_l_2",
    "leg_back_l_3",
)
ACTUATOR_NAMES = JOINT_NAMES
FOOT_BODY_NAMES = (
    "leg_front_r_3",
    "leg_front_l_3",
    "leg_back_r_3",
    "leg_back_l_3",
)
FOOT_SITE_NAMES = tuple(f"{name}_foot_site" for name in FOOT_BODY_NAMES)


@dataclass(frozen=True)
class ModelContract:
    stand_key_id: int
    torso_body_id: int
    torso_geom_id: int
    floor_geom_id: int
    qpos_indices: np.ndarray
    dof_indices: np.ndarray
    actuator_ids: np.ndarray
    foot_geom_ids: np.ndarray
    foot_radii: np.ndarray
    foot_site_ids: np.ndarray
    stand_q: np.ndarray
    ctrl_low: np.ndarray
    ctrl_high: np.ndarray


def resolve_model_contract(model) -> ModelContract:
    """Resolves the stable named Pupper contract while leaving geometry in XML."""

    stand_key_id = _named_id(model, "key", "stand", fallback="home")
    torso_body_id = _named_id(model, "body", "base_link")
    torso_geom_id = _named_id(model, "geom", "base_disk_collision")
    floor_geom_id = _named_id(model, "geom", "floor")
    qpos_indices = np.asarray(
        [model.jnt_qposadr[_named_id(model, "joint", name)] for name in JOINT_NAMES], dtype=np.int32
    )
    dof_indices = np.asarray(
        [model.jnt_dofadr[_named_id(model, "joint", name)] for name in JOINT_NAMES], dtype=np.int32
    )
    actuator_ids = np.asarray([_named_id(model, "actuator", name) for name in ACTUATOR_NAMES], dtype=np.int32)
    foot_site_ids = np.asarray([_named_id(model, "site", name) for name in FOOT_SITE_NAMES], dtype=np.int32)
    foot_geom_ids = np.asarray([_collision_geom_for_body(model, name) for name in FOOT_BODY_NAMES], dtype=np.int32)
    foot_radii = np.asarray(model.geom_size[foot_geom_ids, 0], dtype=np.float64).copy()
    stand_q = np.asarray(model.key_qpos[stand_key_id, qpos_indices], dtype=np.float64).copy()
    ctrl_low = np.asarray(model.actuator_ctrlrange[actuator_ids, 0], dtype=np.float64).copy()
    ctrl_high = np.asarray(model.actuator_ctrlrange[actuator_ids, 1], dtype=np.float64).copy()
    return ModelContract(
        stand_key_id=stand_key_id,
        torso_body_id=torso_body_id,
        torso_geom_id=torso_geom_id,
        floor_geom_id=floor_geom_id,
        qpos_indices=qpos_indices,
        dof_indices=dof_indices,
        actuator_ids=actuator_ids,
        foot_geom_ids=foot_geom_ids,
        foot_radii=foot_radii,
        foot_site_ids=foot_site_ids,
        stand_q=stand_q,
        ctrl_low=ctrl_low,
        ctrl_high=ctrl_high,
    )


def _named_id(model, kind: str, name: str, fallback: str | None = None) -> int:
    accessor = getattr(model, kind)
    try:
        return int(accessor(name).id)
    except KeyError:
        if fallback is not None:
            try:
                return int(accessor(fallback).id)
            except KeyError:
                pass
        suffix = f" or {fallback!r}" if fallback else ""
        raise ValueError(f"Target XML is missing required {kind} {name!r}{suffix}") from None


def _collision_geom_for_body(model, body_name: str) -> int:
    body_id = _named_id(model, "body", body_name)
    candidates = np.flatnonzero(model.geom_bodyid == body_id)
    collidable = [int(i) for i in candidates if model.geom_contype[i] or model.geom_conaffinity[i]]
    if len(collidable) != 1:
        raise ValueError(
            f"Expected one collidable foot geom on body {body_name!r}, found {len(collidable)}"
        )
    return collidable[0]
