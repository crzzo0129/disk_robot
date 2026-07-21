from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from disk_robot.model_paths import ACTIVE_MODEL_XML
from disk_robot.model_contract import FOOT_BODY_NAMES, resolve_model_contract


DEFAULT_XML = ACTIVE_MODEL_XML


def _convex_polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    x = ordered[:, 0]
    y = ordered[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def diagnose(xml_path: Path, keyframe: str) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    contract = resolve_model_contract(model)
    key_id = model.key(keyframe).id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    torso_id = contract.torso_body_id
    foot_ids = contract.foot_geom_ids
    foot_radius = float(model.geom_size[foot_ids[0], 0])
    torso_xy = data.xpos[torso_id, :2].copy()
    torso_z = float(data.xpos[torso_id, 2])
    foot_pos = np.array([data.geom_xpos[geom_id] for geom_id in foot_ids])
    foot_xy = foot_pos[:, :2]
    foot_z = foot_pos[:, 2]
    support_center = foot_xy.mean(axis=0)
    offset = torso_xy - support_center
    support_area = _convex_polygon_area(foot_xy)

    print(f"xml={xml_path}")
    print(f"keyframe={keyframe}")
    print(f"qpos_z={float(data.qpos[2]):.6f}")
    print(f"torso_z={torso_z:.6f}")
    print(f"foot_radius={foot_radius:.6f}")
    print(f"support_center_xy=({support_center[0]:.6f}, {support_center[1]:.6f})")
    print(f"torso_xy=({torso_xy[0]:.6f}, {torso_xy[1]:.6f})")
    print(f"torso_minus_support_center_xy=({offset[0]:.6f}, {offset[1]:.6f}) norm={np.linalg.norm(offset):.6f}")
    print(f"support_polygon_area={support_area:.6f}")
    print("feet:")
    for name, pos in zip(FOOT_BODY_NAMES, foot_pos):
        clearance = float(pos[2] - foot_radius)
        print(
            f"  {name}: x={pos[0]:.6f} y={pos[1]:.6f} z={pos[2]:.6f} "
            f"clearance={clearance:.6f}"
        )

    avg_height_qpos = float(data.qpos[2] + (foot_radius - foot_z.mean()))
    min_height_qpos = float(data.qpos[2] + (foot_radius - foot_z.min()))
    print(f"suggested_qpos_z_avg_feet_on_ground={avg_height_qpos:.6f}")
    print(f"suggested_qpos_z_min_feet_on_ground={min_height_qpos:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose Pupper stand support center and foot clearances.")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--keyframe", default="stand")
    args = parser.parse_args()
    diagnose(args.xml_path, args.keyframe)
