"""Interactively view a disk-robot rolling structure/COM variant."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from disk_robot.structure_variants import StructureVariant, apply_structure_variant
from scripts.sweep_rolling_variants import DEFAULT_XML, _prepare_data, set_disk_com


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--hip-y", type=float, default=0.09)
    parser.add_argument("--leg-scale", type=float, default=0.85)
    parser.add_argument("--disk-radius", type=float, default=0.20)
    parser.add_argument("--com-x", type=float, default=-0.005)
    parser.add_argument("--com-z", type=float, default=0.030)
    parser.add_argument("--initial-speed", type=float, default=0.8)
    parser.add_argument("--direction", choices=("forward", "reverse", "rest"), default="forward")
    parser.add_argument("--duration", type=float, default=8.0, help="Use 0 to run until the viewer closes.")
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--torque-limit", type=float, default=6.0)
    return parser.parse_args(argv)


def build_model(args):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser().resolve()))
    apply_structure_variant(
        model,
        StructureVariant(args.hip_y, args.leg_scale, args.disk_radius),
    )
    set_disk_com(model, args.com_x, args.com_z)
    model.actuator_gainprm[:, 0] = args.kp
    model.actuator_biasprm[:, 1] = -args.kp
    model.actuator_biasprm[:, 2] = -args.kd
    model.actuator_forcerange[:, 0] = -abs(args.torque_limit)
    model.actuator_forcerange[:, 1] = abs(args.torque_limit)
    return model


def set_launch_velocity(model, data, direction, initial_speed):
    free_joint_id = model.joint("world_to_body").id
    dof = int(model.jnt_dofadr[free_joint_id])
    radius = float(model.geom_size[model.geom("base_disk_collision").id, 0])
    sign = {"forward": 1.0, "reverse": -1.0, "rest": 0.0}[direction]
    speed = sign * abs(initial_speed)
    data.qvel[dof] = speed
    data.qvel[dof + 4] = speed / radius
    return dof, radius


def main(argv=None):
    import mujoco
    from mujoco import viewer

    args = parse_args(argv)
    model = build_model(args)
    data = _prepare_data(model)
    dof, radius = set_launch_velocity(model, data, args.direction, args.initial_speed)
    mujoco.mj_forward(model, data)
    start_x = float(data.qpos[0])
    next_status = 0.0
    print(
        f"rolling_view direction={args.direction} hip_y={args.hip_y:.3f} "
        f"leg_scale={args.leg_scale:.2f} radius={args.disk_radius:.3f} "
        f"disk_com=({args.com_x:+.3f},{args.com_z:+.3f}) initial_speed={args.initial_speed:.3f}",
        flush=True,
    )
    with viewer.launch_passive(model, data) as active_viewer:
        active_viewer.cam.trackbodyid = model.body("base_link").id
        active_viewer.cam.distance = 1.1
        active_viewer.cam.azimuth = 135
        active_viewer.cam.elevation = -18
        wall_start = time.perf_counter()
        while active_viewer.is_running() and (args.duration <= 0.0 or data.time < args.duration):
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            active_viewer.sync()
            if data.time >= next_status:
                vx = float(data.qvel[dof])
                wy = float(data.qvel[dof + 4])
                print(
                    f"t={data.time:5.2f} x={data.qpos[0] - start_x:+.3f} "
                    f"y={data.qpos[1]:+.3f} vx={vx:+.3f} wy={wy:+.3f} "
                    f"slip={abs(vx - radius * wy):.3f}",
                    flush=True,
                )
                next_status += 0.5
            delay = model.opt.timestep - (time.perf_counter() - step_start)
            if delay > 0.0:
                time.sleep(delay)
        print(
            f"finished sim_time={data.time:.2f} wall_time={time.perf_counter() - wall_start:.2f} "
            f"distance_x={data.qpos[0] - start_x:+.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
