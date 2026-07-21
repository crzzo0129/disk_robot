"""Screen provisional Pupper structure variants with the shared IK reference."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from disk_robot.ik_reference import IKReferenceSpec, build_ik_reference_from_model
from disk_robot.model_paths import BASE_MODEL_XML
from disk_robot.model_contract import resolve_model_contract
from disk_robot.structure_variants import StructureVariant, apply_structure_variant


# StructureVariant values are absolute relative to the unscaled Pupper model.  Do not
# import the training environment default here: it points at the already-scaled selected
# candidate, which would compound leg_scale during a new sweep.
DEFAULT_XML = BASE_MODEL_XML


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--hip-y", type=float, nargs="+", default=[0.07, 0.085, 0.09])
    parser.add_argument("--leg-scale", type=float, nargs="+", default=[1.0, 0.9, 0.85])
    parser.add_argument("--disk-radius", type=float, nargs="+", default=[0.20, 0.18, 0.17])
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--phase", type=float, default=0.0)
    parser.add_argument("--frequency", type=float, default=0.8)
    parser.add_argument("--stride", type=float, default=0.04)
    parser.add_argument("--height", type=float, default=0.025)
    parser.add_argument("--duty", type=float, default=0.72)
    parser.add_argument("--ramp", type=float, default=0.5)
    parser.add_argument("--kp", type=float, default=10.0)
    parser.add_argument("--kd", type=float, default=0.4)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    parser.add_argument("--command-vx", type=float, default=0.03)
    parser.add_argument("--out", type=Path, default=Path("structure_sweep"))
    return parser.parse_args(argv)


def _interpolate_reference(reference, phase):
    sample = (phase % 1.0) * len(reference)
    lower = int(np.floor(sample))
    upper = (lower + 1) % len(reference)
    alpha = sample - np.floor(sample)
    return reference[lower] + alpha * (reference[upper] - reference[lower])


def _roll_pitch(rotation):
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    return float(roll), float(pitch)


def _disk_floor_contact(model, data, torso_geom_id, floor_geom_id):
    for contact in data.contact:
        if contact.dist > 0.005:
            continue
        pair = {int(contact.geom[0]), int(contact.geom[1])}
        if pair == {torso_geom_id, floor_geom_id}:
            return 1.0
    return 0.0


def _folded_envelope(model, contract):
    import mujoco

    data = mujoco.MjData(model)
    folded_id = model.key("folded").id
    mujoco.mj_resetDataKeyframe(model, data, folded_id)
    mujoco.mj_forward(model, data)
    disk_id = contract.torso_geom_id
    disk_center = data.geom_xpos[disk_id]
    disk_radius = float(model.geom_size[disk_id, 0])
    radial_extents = []
    half_widths = []
    for geom_id, foot_radius in zip(contract.foot_geom_ids, contract.foot_radii):
        offset = data.geom_xpos[geom_id] - disk_center
        radial_extents.append(float(np.linalg.norm(offset[[0, 2]]) + foot_radius))
        half_widths.append(float(abs(offset[1]) + foot_radius))
    return disk_radius - max(radial_extents), max(half_widths)


def simulate_variant(args, variant):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser().resolve()))
    apply_structure_variant(model, variant)
    contract = resolve_model_contract(model)
    ids = contract.actuator_ids
    model.actuator_gainprm[ids, 0] = args.kp
    model.actuator_biasprm[ids, 1] = -args.kp
    model.actuator_biasprm[ids, 2] = -args.kd
    model.actuator_forcerange[ids, 0] = -args.torque_limit
    model.actuator_forcerange[ids, 1] = args.torque_limit
    spec = IKReferenceSpec(
        samples=256,
        frequency=args.frequency,
        stride_length=args.stride,
        step_height=args.height,
        duty=args.duty,
        mode="trot",
    )
    reference = build_ik_reference_from_model(model, spec).joint_targets
    folded_radial_margin, folded_half_width = _folded_envelope(model, contract)

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, contract.stand_key_id)
    data.qpos[contract.qpos_indices] = contract.stand_q
    data.qvel[:] = 0.0
    data.ctrl[ids] = contract.stand_q
    mujoco.mj_forward(model, data)
    foot_bottom = data.geom_xpos[contract.foot_geom_ids, 2] - contract.foot_radii
    data.qpos[2] += 0.001 - float(np.min(foot_bottom))
    mujoco.mj_forward(model, data)

    start_x = float(data.qpos[0])
    control_repeat = 5
    control_dt = model.opt.timestep * control_repeat
    control_steps = max(1, round(args.duration / control_dt))
    values = []
    failed = False
    for step in range(control_steps):
        sim_time = step * control_dt
        target = _interpolate_reference(
            reference,
            args.phase + sim_time * args.frequency,
        )
        blend = 1.0 if args.ramp <= 0.0 else min(1.0, sim_time / args.ramp)
        target = contract.stand_q + blend * (target - contract.stand_q)
        data.ctrl[ids] = np.clip(target, contract.ctrl_low, contract.ctrl_high)
        saturated = []
        for _ in range(control_repeat):
            mujoco.mj_step(model, data)
            saturated.append(np.mean(np.abs(data.actuator_force[ids]) >= 0.99 * args.torque_limit))

        rotation = data.xmat[contract.torso_body_id].reshape(3, 3)
        roll, pitch = _roll_pitch(rotation)
        angular = rotation.T @ data.cvel[contract.torso_body_id, :3]
        tracking = np.sqrt(
            np.mean(np.square(target - data.qpos[contract.qpos_indices]))
        )
        disk_contact = _disk_floor_contact(
            model,
            data,
            contract.torso_geom_id,
            contract.floor_geom_id,
        )
        values.append(
            (
                roll,
                pitch,
                angular[0],
                angular[1],
                tracking,
                np.mean(saturated),
                disk_contact,
            )
        )
        upright = float(rotation[2, 2])
        if data.xpos[contract.torso_body_id, 2] < 0.16 or upright < 0.65:
            failed = True
            break

    values = np.asarray(values)
    elapsed = max(len(values) * control_dt, control_dt)
    distance = float(data.qpos[0] - start_x)
    mean_velocity = distance / elapsed
    velocity_error = abs(mean_velocity - args.command_vx)
    rp_rms = float(np.sqrt(np.mean(values[:, 0] ** 2 + values[:, 1] ** 2)))
    wxy_rms = float(np.sqrt(np.mean(values[:, 2] ** 2 + values[:, 3] ** 2)))
    tracking_rmse = float(np.mean(values[:, 4]))
    saturation = float(np.mean(values[:, 5]))
    disk_contact_fraction = float(np.mean(values[:, 6]))
    score = (
        mean_velocity
        - 0.25 * velocity_error
        - 0.01 * wxy_rms
        - 0.05 * tracking_rmse
        - 0.05 * saturation
        - 0.10 * disk_contact_fraction
        - 0.50 * max(0.0, -folded_radial_margin)
        - (1.0 if failed else 0.0)
    )
    return {
        "hip_y": variant.hip_y,
        "leg_scale": variant.leg_scale,
        "disk_radius": variant.disk_radius,
        "score": score,
        "distance_x": distance,
        "mean_velocity_x": mean_velocity,
        "velocity_error": velocity_error,
        "roll_pitch_rms_deg": float(np.degrees(rp_rms)),
        "roll_pitch_rate_rms": wxy_rms,
        "tracking_rmse": tracking_rmse,
        "torque_saturation_fraction": saturation,
        "disk_contact_fraction": disk_contact_fraction,
        "folded_radial_margin": folded_radial_margin,
        "folded_half_width": folded_half_width,
        "elapsed": elapsed,
        "failed": failed,
    }


def main(argv=None):
    args = parse_args(argv)
    variants = [
        StructureVariant(hip_y=hip_y, leg_scale=leg_scale, disk_radius=disk_radius)
        for hip_y, leg_scale, disk_radius in itertools.product(
            args.hip_y,
            args.leg_scale,
            args.disk_radius,
        )
    ]
    results = []
    for index, variant in enumerate(variants, start=1):
        result = simulate_variant(args, variant)
        results.append(result)
        print(
            f"variant={index}/{len(variants)} hip_y={variant.hip_y:.3f} "
            f"leg_scale={variant.leg_scale:.2f} disk_radius={variant.disk_radius:.3f} "
            f"vx={result['mean_velocity_x']:.4f} rp={result['roll_pitch_rms_deg']:.2f}deg "
            f"track={result['tracking_rmse']:.4f} sat={100 * result['torque_saturation_fraction']:.1f}% "
            f"failed={result['failed']}",
            flush=True,
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.out.with_suffix(".json")
    csv_path = args.out.with_suffix(".csv")
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print("\nTop structure variants")
    for rank, result in enumerate(results[:10], start=1):
        print(
            f"{rank:2d}. hip_y={result['hip_y']:.3f} leg_scale={result['leg_scale']:.2f} "
            f"disk_radius={result['disk_radius']:.3f} score={result['score']:.5f} "
            f"vx={result['mean_velocity_x']:.4f} rp={result['roll_pitch_rms_deg']:.2f}deg"
        )
    print(f"saved_json={json_path.resolve()}")
    print(f"saved_csv={csv_path.resolve()}")


if __name__ == "__main__":
    main()
