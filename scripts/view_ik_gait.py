"""Visualize a foot-space IK gait on the target Pupper model."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    sys.path.remove(str(PROJECT_ROOT))
except ValueError:
    pass
sys.path.insert(0, str(PROJECT_ROOT))

from disk_robot.ik_gait import FootSpaceIKGait, FootTrajectoryParams
from disk_robot.model_contract import resolve_model_contract


DEFAULT_XML = PROJECT_ROOT / "assets" / "pupper_v3_disk_visual.xml"
PUPPER_DEFAULT_POSE = np.array(
    (0.26, 0.0, -0.52, -0.26, 0.0, 0.52, 0.26, 0.0, -0.52, -0.26, 0.0, 0.52),
    dtype=np.float64,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--mode", choices=("crawl", "trot"), default="crawl")
    parser.add_argument("--neutral-pose", choices=("model", "pupper"), default="model")
    parser.add_argument("--frequency", type=float, default=0.8)
    parser.add_argument("--stride", type=float, default=0.04)
    parser.add_argument("--height", type=float, default=0.025)
    parser.add_argument("--duty", type=float, default=0.72)
    parser.add_argument("--ramp", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--realtime", type=float, default=1.0)
    parser.add_argument("--kinematic", action="store_true", help="Show joint motion with the floating base fixed.")
    parser.add_argument("--headless", action="store_true", help="Run physics and print metrics without a viewer.")
    return parser.parse_args(argv)


def _reset(model, data, contract, mujoco) -> None:
    mujoco.mj_resetDataKeyframe(model, data, contract.stand_key_id)
    data.qpos[contract.qpos_indices] = contract.stand_q
    data.qvel[:] = 0.0
    data.ctrl[contract.actuator_ids] = contract.stand_q
    mujoco.mj_forward(model, data)


def _roll_pitch(rotation: np.ndarray) -> tuple[float, float]:
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    return roll, pitch


def _peak_contact_force(model, data, mujoco) -> float:
    peak = 0.0
    force = np.zeros(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        mujoco.mj_contactForce(model, data, contact_index, force)
        peak = max(peak, abs(float(force[0])))
    return peak
    foot_bottom = data.geom_xpos[contract.foot_geom_ids, 2] - contract.foot_radii
    data.qpos[2] += 0.001 - float(np.min(foot_bottom))
    mujoco.mj_forward(model, data)


def main(argv=None):
    args = parse_args(argv)
    if args.frequency <= 0.0 or args.realtime <= 0.0:
        raise SystemExit("--frequency and --realtime must be positive")
    if args.kp < 0.0 or args.kd < 0.0 or args.torque_limit <= 0.0:
        raise SystemExit("--kp and --kd must be nonnegative; --torque-limit must be positive")
    if not 0.5 <= args.duty < 1.0:
        raise SystemExit("--duty must be in [0.5, 1.0)")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    contract = resolve_model_contract(model)
    if args.neutral_pose == "pupper":
        contract = replace(contract, stand_q=PUPPER_DEFAULT_POSE.copy())
    model.actuator_gainprm[contract.actuator_ids, 0] = args.kp
    model.actuator_biasprm[contract.actuator_ids, 1] = -args.kp
    model.actuator_biasprm[contract.actuator_ids, 2] = -args.kd
    model.actuator_forcerange[contract.actuator_ids, 0] = -args.torque_limit
    model.actuator_forcerange[contract.actuator_ids, 1] = args.torque_limit
    params = FootTrajectoryParams(
        frequency=args.frequency,
        stride_length=args.stride,
        step_height=args.height,
        duty=args.duty,
        mode=args.mode,
    )
    gait = FootSpaceIKGait(model, contract, params)
    _reset(model, data, contract, mujoco)
    initial_base_qpos = data.qpos[:7].copy()
    control_repeat = 5
    control_dt = model.opt.timestep * control_repeat
    state = {"paused": False, "reset": False}

    def key_callback(keycode):
        if keycode == ord(" "):
            state["paused"] = not state["paused"]
        elif keycode in (ord("R"), ord("r")):
            state["reset"] = True

    print(
        f"model={args.xml.resolve()} mode={args.mode} neutral={args.neutral_pose} kinematic={args.kinematic} "
        f"kp={args.kp:g} kd={args.kd:g} torque_limit={args.torque_limit:g}"
    )
    print("Space=pause; R=reset; close window=exit")
    print("Metrics are one-second averages in the torso frame.")

    if args.headless:
        viewer_context = nullcontext(None)
    else:
        from mujoco import viewer

        viewer_context = viewer.launch_passive(model, data, key_callback=key_callback)

    with viewer_context as window:
        if window is not None:
            window.cam.azimuth = 135
            window.cam.elevation = -18
            window.cam.distance = 1.1
        step_count = 0
        interval = []
        while window is None or window.is_running():
            if state["reset"]:
                _reset(model, data, contract, mujoco)
                gait.reset()
                initial_base_qpos = data.qpos[:7].copy()
                step_count = 0
                interval.clear()
                state["reset"] = False

            sim_time = step_count * control_dt
            if args.duration > 0.0 and sim_time >= args.duration:
                break
            wall_start = time.perf_counter()

            if not state["paused"]:
                target = gait.targets(sim_time)
                blend = 1.0 if args.ramp <= 0.0 else min(1.0, sim_time / args.ramp)
                target = contract.stand_q + blend * (target - contract.stand_q)
                old_pos = data.xpos[contract.torso_body_id].copy()
                if args.kinematic:
                    data.qpos[:7] = initial_base_qpos
                    data.qpos[contract.qpos_indices] = target
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    saturation = 0.0
                    contact_peak = 0.0
                else:
                    data.ctrl[contract.actuator_ids] = np.clip(target, contract.ctrl_low, contract.ctrl_high)
                    saturated_substeps = []
                    contact_peak = 0.0
                    for _ in range(control_repeat):
                        mujoco.mj_step(model, data)
                        saturated_substeps.append(
                            np.mean(
                                np.abs(data.actuator_force[contract.actuator_ids])
                                >= 0.99 * args.torque_limit
                            )
                        )
                        contact_peak = max(contact_peak, _peak_contact_force(model, data, mujoco))
                    saturation = float(np.mean(saturated_substeps))
                step_count += 1

                torso_rot = data.xmat[contract.torso_body_id].reshape(3, 3)
                velocity = torso_rot.T @ ((data.xpos[contract.torso_body_id] - old_pos) / control_dt)
                angular = torso_rot.T @ data.cvel[contract.torso_body_id, :3]
                roll, pitch = _roll_pitch(torso_rot)
                tracking_rmse = float(np.sqrt(np.mean(np.square(target - data.qpos[contract.qpos_indices]))))
                interval.append(
                    (
                        velocity[0],
                        velocity[1],
                        angular[2],
                        torso_rot[2, 2],
                        tracking_rmse,
                        saturation,
                        roll * roll + pitch * pitch,
                        angular[0] * angular[0] + angular[1] * angular[1],
                        contact_peak,
                    )
                )
                if step_count % max(1, round(1.0 / control_dt)) == 0:
                    mean = np.mean(np.asarray(interval), axis=0)
                    print(
                        f"t={step_count * control_dt:6.2f}s  vx={mean[0]: .3f}  vy={mean[1]: .3f}  "
                        f"wz={mean[2]: .3f}  upright={mean[3]:.3f}  track={mean[4]:.3f}  "
                        f"sat={100.0 * mean[5]:4.1f}%  rp={np.degrees(np.sqrt(mean[6])):4.1f}deg  "
                        f"wxy={np.sqrt(mean[7]):.3f}  impact={mean[8]:.1f}N  "
                        f"ik_max={np.max(gait.last_errors):.5f}"
                    )
                    interval.clear()

            if window is not None:
                window.cam.lookat[:] = data.xpos[contract.torso_body_id]
                window.sync()
            if window is None:
                continue
            if state["paused"]:
                time.sleep(0.02)
            else:
                elapsed = time.perf_counter() - wall_start
                time.sleep(max(0.0, control_dt / args.realtime - elapsed))


if __name__ == "__main__":
    main()
