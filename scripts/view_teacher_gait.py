"""Visualize the verified open-loop teacher on the target Pupper model."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

# Support both `python -m scripts.view_teacher_gait` and direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(PROJECT_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(PROJECT_ROOT))

from disk_robot.walk_config import WalkTaskConfig
from disk_robot.walk_env import DiskRobotWalkEnv
from disk_robot.model_paths import ACTIVE_MODEL_XML


DEFAULT_XML = ACTIVE_MODEL_XML


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation seconds; <= 0 runs until closed.")
    parser.add_argument("--realtime", type=float, default=1.0, help="Playback speed relative to real time.")
    parser.add_argument("--command-vx", type=float, default=0.15)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.realtime <= 0.0:
        raise SystemExit("--realtime must be positive")

    from mujoco import viewer

    config = replace(
        WalkTaskConfig(),
        max_episode_steps=1_000_000,
        reset_joint_noise=0.0,
        reset_height_noise=0.0,
        command_vx_min=args.command_vx,
        command_vx_max=args.command_vx,
        command_zero_probability=0.0,
        teacher_blend=1.0,
    )
    env = DiskRobotWalkEnv(config=config, xml_path=args.xml.resolve(), seed=0)
    env.reset()
    action = np.zeros(config.action_size, dtype=np.float64)
    control_dt = env.model.opt.timestep * max(1, config.action_repeat)
    state = {"paused": False, "reset": False}

    def key_callback(keycode):
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
        elif keycode in (ord("R"), ord("r")):
            state["reset"] = True

    print(f"model={args.xml.resolve()}")
    print("teacher=PUPPER_FORWARD_TEACHER; Space=pause; R=reset; close window=exit")
    print("Metrics are body-frame averages over each one-second interval.")

    with viewer.launch_passive(env.model, env.data, key_callback=key_callback) as window:
        window.cam.azimuth = 135
        window.cam.elevation = -18
        window.cam.distance = 1.1
        window.cam.lookat[:] = env.data.xpos[env.contract.torso_body_id]

        next_report = control_dt
        interval = []
        while window.is_running():
            if state["reset"]:
                env.reset()
                next_report = control_dt
                interval.clear()
                state["reset"] = False

            sim_time = env.step_count * control_dt
            if args.duration > 0.0 and sim_time >= args.duration:
                break

            wall_start = time.perf_counter()
            if not state["paused"]:
                _, _, terminated, _, info = env.step(action)
                interval.append(
                    (
                        info["velocity_x"],
                        info["velocity_y"],
                        info["yaw_rate"],
                        info["upright"],
                        info["disk_contact_count"],
                    )
                )
                sim_time = env.step_count * control_dt
                if sim_time >= next_report and interval:
                    mean = np.mean(np.asarray(interval, dtype=np.float64), axis=0)
                    print(
                        f"t={sim_time:6.2f}s  vx={mean[0]: .3f}  vy={mean[1]: .3f}  "
                        f"wz={mean[2]: .3f}  upright={mean[3]:.3f}  disk_contacts={mean[4]:.2f}"
                    )
                    interval.clear()
                    next_report += 1.0
                if terminated:
                    print(f"terminated at t={sim_time:.2f}s; press R to restart")
                    state["paused"] = True

            window.cam.lookat[:] = env.data.xpos[env.contract.torso_body_id]
            window.sync()
            if not state["paused"]:
                elapsed = time.perf_counter() - wall_start
                time.sleep(max(0.0, control_dt / args.realtime - elapsed))
            else:
                time.sleep(0.02)


if __name__ == "__main__":
    main()
