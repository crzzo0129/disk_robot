from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from disk_robot.gait import GaitParams, make_open_loop_targets
from disk_robot.walk_config import ACTUATOR_NAMES, FOOT_GEOMS, JOINT_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_ROOT / "assets" / "disk_quadruped_extreme_train.xml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a simple open-loop gait on the disk robot MuJoCo model.")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--keyframe", default="walk_stand")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument("--frequency", type=float, default=1.0, help="Gait cycle frequency in Hz.")
    parser.add_argument("--hip-amplitude", type=float, default=0.10)
    parser.add_argument("--hip-stance-amplitude", type=float, default=None)
    parser.add_argument("--hip-swing-amplitude", type=float, default=None)
    parser.add_argument("--march-hip-compensation", type=float, default=0.0)
    parser.add_argument("--knee-amplitude", type=float, default=0.22)
    parser.add_argument("--knee-lift-amplitude", type=float, default=None)
    parser.add_argument("--abd-amplitude", type=float, default=0.0)
    parser.add_argument("--direction", type=float, default=-1.0)
    parser.add_argument("--front-knee-sign", type=float, default=-1.0)
    parser.add_argument("--hind-knee-sign", type=float, default=-1.0)
    parser.add_argument("--duty", type=float, default=0.6, help="Fraction of cycle spent in stance.")
    parser.add_argument("--mode", choices=["trot", "pace", "bound", "crawl", "march"], default="trot")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--render", type=Path, default=None, help="Save an offscreen MP4 instead of opening viewer.")
    parser.add_argument("--mujoco-gl", default="egl", help="OpenGL backend for offscreen rendering.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--camera", default="side_cam")
    parser.add_argument("--print-every", type=float, default=0.5)
    return parser.parse_args(argv)


def _ids_by_name(model, mujoco, names, obj_type):
    ids = []
    for name in names:
        obj_id = mujoco.mj_name2id(model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Object not found: {name}")
        ids.append(obj_id)
    return np.array(ids, dtype=np.int32)


def _gait_params_from_args(args):
    return GaitParams(
        frequency=args.frequency,
        hip_stance_amplitude=args.hip_amplitude if args.hip_stance_amplitude is None else args.hip_stance_amplitude,
        hip_swing_amplitude=args.hip_amplitude if args.hip_swing_amplitude is None else args.hip_swing_amplitude,
        knee_lift_amplitude=args.knee_amplitude if args.knee_lift_amplitude is None else args.knee_lift_amplitude,
        abd_amplitude=args.abd_amplitude,
        duty=args.duty,
        mode=args.mode,
        direction=args.direction,
        front_knee_sign=args.front_knee_sign,
        hind_knee_sign=args.hind_knee_sign,
        march_hip_compensation=args.march_hip_compensation,
    )


def make_targets(neutral, t, args):
    return make_open_loop_targets(neutral, t, _gait_params_from_args(args))


def main(argv=None):
    args = parse_args(argv)
    if args.render is not None and args.mujoco_gl:
        os.environ.setdefault("MUJOCO_GL", args.mujoco_gl)
        os.environ.setdefault("PYOPENGL_PLATFORM", args.mujoco_gl)

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.xml_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if key_id < 0:
        raise ValueError(f"Keyframe not found: {args.keyframe}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    actuator_ids = _ids_by_name(model, mujoco, ACTUATOR_NAMES, mujoco.mjtObj.mjOBJ_ACTUATOR)
    qpos_indices = _ids_by_name(model, mujoco, JOINT_NAMES, mujoco.mjtObj.mjOBJ_JOINT)
    qpos_indices = np.array([model.jnt_qposadr[joint_id] for joint_id in qpos_indices], dtype=np.int32)
    foot_ids = _ids_by_name(model, mujoco, FOOT_GEOMS, mujoco.mjtObj.mjOBJ_GEOM)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "disk_torso")

    neutral = data.qpos[qpos_indices].copy()
    ctrl_low = model.actuator_ctrlrange[actuator_ids, 0]
    ctrl_high = model.actuator_ctrlrange[actuator_ids, 1]
    data.ctrl[actuator_ids] = neutral

    def step_controller(sim_time):
        targets = make_targets(neutral, sim_time, args)
        targets = np.clip(targets, ctrl_low, ctrl_high)
        data.ctrl[actuator_ids] = targets

    def print_status(start_x, next_print):
        torso = data.xpos[torso_id]
        foot_z = [float(data.geom_xpos[foot_id][2]) for foot_id in foot_ids]
        print(
            f"t={data.time:.2f} x={torso[0]:.3f} dx={torso[0] - start_x:.3f} "
            f"y={torso[1]:.3f} z={torso[2]:.3f} min_foot_z={min(foot_z):.3f}",
            flush=True,
        )
        return next_print + args.print_every

    for _ in range(int(args.settle / model.opt.timestep)):
        mujoco.mj_step(model, data)

    start_x = float(data.xpos[torso_id][0])
    next_print = data.time

    if args.render is not None:
        import imageio.v3 as iio

        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        frames = []
        render_dt = 1.0 / max(args.fps, 1)
        next_frame_time = data.time
        camera = args.camera if args.camera else None
        end_time = data.time + args.duration
        while data.time < end_time:
            step_controller(data.time)
            mujoco.mj_step(model, data)
            if data.time >= next_print:
                next_print = print_status(start_x, next_print)
            if data.time >= next_frame_time:
                try:
                    renderer.update_scene(data, camera=camera)
                except ValueError:
                    renderer.update_scene(data)
                frames.append(renderer.render())
                next_frame_time += render_dt
        renderer.close()
        args.render.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.render, frames, fps=args.fps)
        print(f"video_saved path={args.render} frames={len(frames)} fps={args.fps}", flush=True)
    elif args.viewer:
        from mujoco import viewer

        with viewer.launch_passive(model, data) as window:
            if args.camera:
                try:
                    window.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
                    window.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                except Exception:
                    pass
            end_time = data.time + args.duration
            while window.is_running() and data.time < end_time:
                step_controller(data.time)
                mujoco.mj_step(model, data)
                if data.time >= next_print:
                    next_print = print_status(start_x, next_print)
                window.sync()
                time.sleep(model.opt.timestep)
    else:
        end_time = data.time + args.duration
        while data.time < end_time:
            step_controller(data.time)
            mujoco.mj_step(model, data)
            if data.time >= next_print:
                next_print = print_status(start_x, next_print)

    torso = data.xpos[torso_id]
    print(
        f"summary duration={args.duration:.2f} mode={args.mode} frequency={args.frequency:.2f} "
        f"dx={torso[0] - start_x:.3f} dy={torso[1]:.3f} final_z={torso[2]:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
