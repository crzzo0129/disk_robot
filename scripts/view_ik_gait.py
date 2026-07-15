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
from disk_robot.ik_reference import IKReferenceSpec, build_ik_reference_from_model
from disk_robot.gait_speed import MAX_CALIBRATED_FORWARD_SPEED, plan_forward_gait
from disk_robot.model_contract import resolve_model_contract
from disk_robot.structure_variants import StructureVariant, apply_structure_variant
from disk_robot.video_recorder import MujocoVideoRecorder


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
    parser.add_argument(
        "--target-speed",
        type=float,
        default=None,
        metavar="M_PER_S",
        help=(
            "Select calibrated frequency/stride for the candidate structure; "
            "overrides --frequency and --stride."
        ),
    )
    parser.add_argument("--height", type=float, default=0.025)
    parser.add_argument("--duty", type=float, default=0.72)
    parser.add_argument("--ramp", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--kd", type=float, default=0.1)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--realtime", type=float, default=1.0)
    parser.add_argument(
        "--training-reference",
        action="store_true",
        help="Use the same warmed 256-sample IK lookup table as teacher training.",
    )
    parser.add_argument("--phase", type=float, default=0.0, help="Initial gait phase in cycles.")
    parser.add_argument("--hip-y", type=float, default=None)
    parser.add_argument("--leg-scale", type=float, default=1.0)
    parser.add_argument("--disk-radius", type=float, default=None)
    parser.add_argument("--kinematic", action="store_true", help="Show joint motion with the floating base fixed.")
    parser.add_argument("--headless", action="store_true", help="Run physics and print metrics without a viewer.")
    parser.add_argument("--video", type=Path, help="Write an MP4 instead of opening the interactive viewer.")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    return parser.parse_args(argv)


def _reset(model, data, contract, mujoco) -> None:
    mujoco.mj_resetDataKeyframe(model, data, contract.stand_key_id)
    data.qpos[contract.qpos_indices] = contract.stand_q
    data.qvel[:] = 0.0
    data.ctrl[contract.actuator_ids] = contract.stand_q
    mujoco.mj_forward(model, data)
    foot_bottom = data.geom_xpos[contract.foot_geom_ids, 2] - contract.foot_radii
    data.qpos[2] += 0.001 - float(np.min(foot_bottom))
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


def main(argv=None):
    args = parse_args(argv)
    try:
        speed_plan = None if args.target_speed is None else plan_forward_gait(args.target_speed)
    except ValueError as exc:
        raise SystemExit(f"invalid --target-speed: {exc}") from exc
    frequency = args.frequency if speed_plan is None else speed_plan.frequency
    stride = args.stride if speed_plan is None else speed_plan.stride_length
    height = args.height if speed_plan is None else args.height * speed_plan.motion_scale
    if frequency <= 0.0 or args.realtime <= 0.0:
        raise SystemExit("--frequency and --realtime must be positive")
    if args.kp < 0.0 or args.kd < 0.0 or args.torque_limit <= 0.0:
        raise SystemExit("--kp and --kd must be nonnegative; --torque-limit must be positive")
    if not 0.5 <= args.duty < 1.0:
        raise SystemExit("--duty must be in [0.5, 1.0)")
    if args.video_fps <= 0 or args.video_width <= 0 or args.video_height <= 0:
        raise SystemExit("video fps and dimensions must be positive")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    current_hip_y = abs(float(model.body_pos[model.body("leg_front_r_1").id, 1]))
    current_disk_radius = float(model.geom_size[model.geom("base_disk_collision").id, 0])
    selected_hip_y = current_hip_y if args.hip_y is None else args.hip_y
    selected_disk_radius = current_disk_radius if args.disk_radius is None else args.disk_radius
    if args.hip_y is not None or args.leg_scale != 1.0 or args.disk_radius is not None:
        apply_structure_variant(
            model,
            StructureVariant(
                hip_y=selected_hip_y,
                leg_scale=args.leg_scale,
                disk_radius=selected_disk_radius,
            ),
        )
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
        frequency=frequency,
        stride_length=stride,
        step_height=height,
        duty=args.duty,
        mode=args.mode,
    )
    gait = FootSpaceIKGait(model, contract, params)
    reference = None
    if args.training_reference:
        reference = build_ik_reference_from_model(
            model,
            IKReferenceSpec(
                samples=256,
                frequency=frequency,
                stride_length=stride,
                step_height=height,
                duty=args.duty,
                mode=args.mode,
            ),
        ).joint_targets
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
        f"kp={args.kp:g} kd={args.kd:g} torque_limit={args.torque_limit:g} "
        f"hip_y={selected_hip_y:g} leg_scale={args.leg_scale:g} "
        f"disk_radius={selected_disk_radius:g} frequency={frequency:g} stride={stride:g}"
    )
    if speed_plan is not None:
        print(
            f"speed_plan target={speed_plan.target_speed:.4f}m/s "
            f"calibrated_limit={MAX_CALIBRATED_FORWARD_SPEED:.4f}m/s"
        )
    print("Space=pause; R=reset; close window=exit")
    print("Metrics are one-second averages in the torso frame.")

    if args.headless or args.video is not None:
        viewer_context = nullcontext(None)
    else:
        from mujoco import viewer

        viewer_context = viewer.launch_passive(model, data, key_callback=key_callback)

    video_context = (
        MujocoVideoRecorder(
            model,
            args.video,
            contract.torso_body_id,
            fps=args.video_fps,
            width=args.video_width,
            height=args.video_height,
            azimuth=90,
            elevation=-10,
            distance=1.2,
        )
        if args.video is not None
        else nullcontext(None)
    )
    with viewer_context as window, video_context as recorder:
        if window is not None:
            window.cam.azimuth = 135
            window.cam.elevation = -18
            window.cam.distance = 1.1
        step_count = 0
        interval = []
        if recorder is not None:
            recorder.capture(data)
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
                if reference is None:
                    target = gait.targets(sim_time)
                    ik_max = np.max(gait.last_errors)
                else:
                    sample = ((args.phase + sim_time * frequency) % 1.0) * len(reference)
                    lower = int(np.floor(sample))
                    upper = (lower + 1) % len(reference)
                    alpha = sample - np.floor(sample)
                    target = reference[lower] + alpha * (reference[upper] - reference[lower])
                    ik_max = float("nan")
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
                world_velocity = (data.xpos[contract.torso_body_id] - old_pos) / control_dt
                body_velocity = torso_rot.T @ world_velocity
                angular = torso_rot.T @ data.cvel[contract.torso_body_id, :3]
                roll, pitch = _roll_pitch(torso_rot)
                tracking_rmse = float(np.sqrt(np.mean(np.square(target - data.qpos[contract.qpos_indices]))))
                interval.append(
                    (
                        world_velocity[0],
                        body_velocity[0],
                        body_velocity[1],
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
                        f"t={step_count * control_dt:6.2f}s  vx_world={mean[0]: .3f}  "
                        f"vx_body={mean[1]: .3f}  vy_body={mean[2]: .3f}  "
                        f"dx={data.qpos[0] - initial_base_qpos[0]: .3f}  wz={mean[3]: .3f}  "
                        f"upright={mean[4]:.3f}  track={mean[5]:.3f}  "
                        f"sat={100.0 * mean[6]:4.1f}%  rp={np.degrees(np.sqrt(mean[7])):4.1f}deg  "
                        f"wxy={np.sqrt(mean[8]):.3f}  impact={mean[9]:.1f}N  "
                        f"ik_max={ik_max:.5f}"
                    )
                    interval.clear()

            if window is not None:
                window.cam.lookat[:] = data.xpos[contract.torso_body_id]
                window.sync()
            if recorder is not None:
                recorder.capture(data)
            if window is None:
                continue
            if state["paused"]:
                time.sleep(0.02)
            else:
                elapsed = time.perf_counter() - wall_start
                time.sleep(max(0.0, control_dt / args.realtime - elapsed))
        if recorder is not None:
            print(f"video={recorder.output_path} frames={recorder.frame_count}")


if __name__ == "__main__":
    main()
