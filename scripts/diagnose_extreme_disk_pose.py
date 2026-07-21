from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from disk_robot.model_paths import LEGACY_EXTREME_XML

DEFAULT_XML = LEGACY_EXTREME_XML
KEYFRAME_NAMES = ("stand", "folded")
FOOT_GEOMS = ("fl_foot", "fr_foot", "hl_foot", "hr_foot")
JOINT_NAMES = (
    "fl_hip_abd",
    "fl_hip_flex",
    "fl_knee",
    "fr_hip_abd",
    "fr_hip_flex",
    "fr_knee",
    "hl_hip_abd",
    "hl_hip_flex",
    "hl_knee",
    "hr_hip_abd",
    "hr_hip_flex",
    "hr_knee",
)


def format_scalar(name: str, value: float) -> str:
    return f"{name}={value:.4f}"


def expand_keyframes(selection: str) -> tuple[str, ...]:
    if selection == "all":
        return KEYFRAME_NAMES
    return (selection,)


def _joint_qpos(model, data, joint_name: str) -> float:
    joint_id = model.joint(joint_name).id
    qpos_addr = model.jnt_qposadr[joint_id]
    return float(data.qpos[qpos_addr])


def _actuator_ctrl(model, data, joint_name: str) -> float:
    actuator_name = f"{joint_name}_act"
    actuator_id = model.actuator(actuator_name).id
    return float(data.ctrl[actuator_id])


def _contact_names(model, data) -> list[str]:
    pairs: list[str] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        geom_1 = model.geom(int(contact.geom1)).name
        geom_2 = model.geom(int(contact.geom2)).name
        pairs.append(f"{geom_1}<->{geom_2}")
    return pairs


def _foot_lines(model, data) -> Iterable[str]:
    for geom_name in FOOT_GEOMS:
        geom_id = model.geom(geom_name).id
        center_z = float(data.geom_xpos[geom_id][2])
        radius = float(model.geom_size[geom_id][0])
        clearance = center_z - radius
        yield (
            f"  {geom_name}: "
            f"{format_scalar('center_z', center_z)}, "
            f"{format_scalar('clearance', clearance)}"
        )


def _joint_lines(model, data) -> Iterable[str]:
    for joint_name in JOINT_NAMES:
        qpos = _joint_qpos(model, data, joint_name)
        ctrl = _actuator_ctrl(model, data, joint_name)
        yield (
            f"  {joint_name}: "
            f"{format_scalar('qpos', qpos)}, "
            f"{format_scalar('ctrl', ctrl)}"
        )


def diagnose_keyframe(model, data, mujoco, keyframe_name: str, settle_steps: int) -> str:
    key_id = model.key(keyframe_name).id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    torso_id = model.body("disk_torso").id
    torso_pos = data.xpos[torso_id]
    torso_quat = data.xquat[torso_id]

    lines = [
        f"[{keyframe_name}]",
        (
            "torso: "
            f"{format_scalar('x', float(torso_pos[0]))}, "
            f"{format_scalar('y', float(torso_pos[1]))}, "
            f"{format_scalar('z', float(torso_pos[2]))}, "
            "quat="
            + " ".join(f"{float(value):.4f}" for value in torso_quat)
        ),
        f"contacts: ncon={data.ncon}",
    ]
    lines.extend(f"  {pair}" for pair in _contact_names(model, data))
    lines.append("feet:")
    lines.extend(_foot_lines(model, data))
    lines.append("joints:")
    lines.extend(_joint_lines(model, data))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print first-frame diagnostics for the extreme disk quadruped keyframes."
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--keyframe",
        choices=("all", *KEYFRAME_NAMES),
        default="all",
        help="Keyframe to inspect. Use 'all' to inspect stand and folded.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=0,
        help="Optional physics steps after reset. 0 reports the exact keyframe frame.",
    )
    return parser.parse_args()


def main() -> None:
    import mujoco

    args = parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    reports = [
        diagnose_keyframe(model, data, mujoco, name, args.settle_steps)
        for name in expand_keyframes(args.keyframe)
    ]
    print("\n\n".join(reports))


if __name__ == "__main__":
    main()
