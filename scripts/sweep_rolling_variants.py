"""Screen disk-robot structure and disk-COM variants for passive rolling quality."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from disk_robot.model_paths import BASE_MODEL_XML
from disk_robot.structure_variants import StructureVariant, apply_structure_variant


DEFAULT_XML = BASE_MODEL_XML


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--hip-y", type=float, nargs="+", default=[0.07, 0.09])
    parser.add_argument("--leg-scale", type=float, nargs="+", default=[1.0, 0.85])
    parser.add_argument("--disk-radius", type=float, nargs="+", default=[0.17, 0.20])
    parser.add_argument("--com-x", type=float, nargs="+", default=[-0.03, 0.0, 0.03])
    parser.add_argument("--com-z", type=float, nargs="+", default=[-0.03, 0.0, 0.03])
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--rest-duration", type=float, default=2.0)
    parser.add_argument(
        "--initial-speed",
        type=float,
        default=0.8,
        help="Common launch rim/linear speed; angular speed is computed as v/r.",
    )
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=1.0)
    parser.add_argument("--torque-limit", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=Path("rolling_sweep"))
    return parser.parse_args(argv)


def set_disk_com(model, com_x: float, com_z: float) -> None:
    """Place the base-link inertial COM relative to the disk geometry center."""

    import mujoco

    base_id = model.body("base_link").id
    disk_id = model.geom("base_disk_collision").id
    disk_center = model.geom_pos[disk_id]
    model.body_ipos[base_id] = disk_center + np.array([com_x, 0.0, com_z])
    mujoco.mj_setConst(model, mujoco.MjData(model))


def _contact_flags(model, data, disk_id, floor_id, foot_ids):
    disk_floor = False
    foot_floor = False
    for contact in data.contact:
        if contact.dist > 0.005:
            continue
        pair = {int(contact.geom[0]), int(contact.geom[1])}
        disk_floor |= pair == {disk_id, floor_id}
        foot_floor |= floor_id in pair and bool(pair.intersection(foot_ids))
    return float(disk_floor), float(foot_floor)


def _axis_tilt(model, data, disk_id):
    # A MuJoCo cylinder's local Z axis is its symmetry/rolling axis.  The XML rotates
    # that axis onto body/world Y in the nominal rolling configuration.
    rotation = data.geom_xmat[disk_id].reshape(3, 3)
    alignment = np.clip(abs(float(rotation[1, 2])), 0.0, 1.0)
    return math.acos(alignment)


def _foot_geom_ids(model):
    import mujoco

    foot_ids = set()
    for name in ("leg_front_r_3", "leg_front_l_3", "leg_back_r_3", "leg_back_l_3"):
        body_id = model.body(name).id
        start = int(model.body_geomadr[body_id])
        stop = start + int(model.body_geomnum[body_id])
        for geom_id in range(start, stop):
            # Collision geoms in this model use contype=0, conaffinity=1 and are
            # contacted by the floor's contype=1.  The distal body has one sphere
            # (the foot) plus a non-colliding visual mesh.
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE:
                foot_ids.add(geom_id)
    return foot_ids


def _prepare_data(model):
    import mujoco

    data = mujoco.MjData(model)
    key_id = model.key("rolling_folded").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.ctrl[:] = model.key_ctrl[key_id]
    mujoco.mj_forward(model, data)
    disk_id = model.geom("base_disk_collision").id
    disk_radius = float(model.geom_size[disk_id, 0])
    disk_bottom = float(data.geom_xpos[disk_id, 2]) - disk_radius
    data.qpos[2] += 0.0005 - disk_bottom
    mujoco.mj_forward(model, data)
    return data


def initial_geometry_metrics(model):
    """Report complete-robot COM and foot envelope in the rolling keyframe."""

    data = _prepare_data(model)
    base_id = model.body("base_link").id
    disk_id = model.geom("base_disk_collision").id
    disk_center = data.geom_xpos[disk_id]
    complete_com = data.subtree_com[base_id] - disk_center
    radial_extents = []
    for geom_id in _foot_geom_ids(model):
        offset = data.geom_xpos[geom_id] - disk_center
        radial_extents.append(float(np.linalg.norm(offset[[0, 2]]) + model.geom_size[geom_id, 0]))
    radius = float(model.geom_size[disk_id, 0])
    return {
        "complete_com_x": float(complete_com[0]),
        "complete_com_z": float(complete_com[2]),
        "complete_com_radial_offset": float(np.linalg.norm(complete_com[[0, 2]])),
        "rolling_foot_radial_margin": radius - max(radial_extents),
    }


def _run_trial(model, duration: float, signed_omega: float):
    import mujoco

    data = _prepare_data(model)
    base_id = model.body("base_link").id
    disk_id = model.geom("base_disk_collision").id
    floor_id = model.geom("floor").id
    foot_ids = _foot_geom_ids(model)

    free_joint_id = model.joint("world_to_body").id
    dof = int(model.jnt_dofadr[free_joint_id])
    radius = float(model.geom_size[disk_id, 0])
    direction = 0.0 if signed_omega == 0.0 else math.copysign(1.0, signed_omega)
    data.qvel[dof] = radius * signed_omega
    data.qvel[dof + 4] = signed_omega
    mujoco.mj_forward(model, data)

    start_x = float(data.qpos[0])
    start_y = float(data.qpos[1])
    steps = max(1, round(duration / model.opt.timestep))
    disk_contacts = []
    foot_contacts = []
    slip = []
    tilts = []
    actuator_work = 0.0
    failed = False
    for _ in range(steps):
        mujoco.mj_step(model, data)
        disk_contact, foot_contact = _contact_flags(model, data, disk_id, floor_id, foot_ids)
        disk_contacts.append(disk_contact)
        foot_contacts.append(foot_contact)
        slip.append(abs(float(data.qvel[dof]) - radius * float(data.qvel[dof + 4])))
        tilt = _axis_tilt(model, data, disk_id)
        tilts.append(tilt)
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            joint_dof = int(model.jnt_dofadr[joint_id])
            actuator_work += abs(float(data.actuator_force[actuator_id] * data.qvel[joint_dof])) * model.opt.timestep
        if tilt > math.radians(35.0) or abs(float(data.qpos[1]) - start_y) > 0.20:
            failed = True
            break

    raw_distance = float(data.qpos[0]) - start_x
    directional_distance = raw_distance if direction == 0.0 else direction * raw_distance
    directional_final_speed = float(data.qvel[dof]) if direction == 0.0 else direction * float(data.qvel[dof])
    return {
        "distance": directional_distance,
        "raw_distance": raw_distance,
        "final_speed": directional_final_speed,
        "lateral_drift": abs(float(data.qpos[1]) - start_y),
        "slip_rms": float(np.sqrt(np.mean(np.square(slip)))),
        "axis_tilt_rms_deg": float(np.degrees(np.sqrt(np.mean(np.square(tilts))))),
        "axis_tilt_max_deg": float(np.degrees(np.max(tilts))),
        "disk_contact_fraction": float(np.mean(disk_contacts)),
        "foot_contact_fraction": float(np.mean(foot_contacts)),
        "actuator_work": actuator_work,
        "elapsed": len(slip) * model.opt.timestep,
        "failed": failed,
    }


def score_result(forward, reverse, rest):
    mean_distance = 0.5 * (forward["distance"] + reverse["distance"])
    direction_asymmetry = abs(forward["distance"] - reverse["distance"])
    mean_slip = 0.5 * (forward["slip_rms"] + reverse["slip_rms"])
    mean_lateral = 0.5 * (forward["lateral_drift"] + reverse["lateral_drift"])
    mean_tilt_deg = 0.5 * (forward["axis_tilt_rms_deg"] + reverse["axis_tilt_rms_deg"])
    mean_disk_contact = 0.5 * (
        forward["disk_contact_fraction"] + reverse["disk_contact_fraction"]
    )
    mean_foot_contact = 0.5 * (
        forward["foot_contact_fraction"] + reverse["foot_contact_fraction"]
    )
    mean_work = 0.5 * (forward["actuator_work"] + reverse["actuator_work"])
    failed = forward["failed"] or reverse["failed"] or rest["failed"]
    score = (
        mean_distance
        - 0.50 * direction_asymmetry
        - 0.50 * mean_slip
        - 0.50 * mean_lateral
        - 0.002 * mean_tilt_deg
        - 0.10 * (1.0 - mean_disk_contact)
        - 0.20 * mean_foot_contact
        # Work is recorded as a useful hardware metric, but only weakly weighted here:
        # absolute joint work is large for a stiff position hold and must not dominate
        # the rolling/contact measurements.
        - 0.0002 * mean_work
        - 0.50 * abs(rest["raw_distance"])
        - (2.0 if failed else 0.0)
    )
    return {
        "score": score,
        "mean_coast_distance": mean_distance,
        "direction_asymmetry": direction_asymmetry,
        "mean_final_speed": 0.5 * (forward["final_speed"] + reverse["final_speed"]),
        "mean_slip_rms": mean_slip,
        "mean_lateral_drift": mean_lateral,
        "mean_axis_tilt_rms_deg": mean_tilt_deg,
        "mean_disk_contact_fraction": mean_disk_contact,
        "mean_foot_contact_fraction": mean_foot_contact,
        "mean_actuator_work": mean_work,
        "rest_drift": rest["raw_distance"],
        "failed": failed,
    }


def simulate_variant(args, variant, com_x, com_z):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.xml.expanduser().resolve()))
    apply_structure_variant(model, variant)
    set_disk_com(model, com_x, com_z)
    model.actuator_gainprm[:, 0] = args.kp
    model.actuator_biasprm[:, 1] = -args.kp
    model.actuator_biasprm[:, 2] = -args.kd
    model.actuator_forcerange[:, 0] = -args.torque_limit
    model.actuator_forcerange[:, 1] = args.torque_limit
    geometry = initial_geometry_metrics(model)
    omega = abs(args.initial_speed) / variant.disk_radius
    forward = _run_trial(model, args.duration, omega)
    reverse = _run_trial(model, args.duration, -omega)
    rest = _run_trial(model, args.rest_duration, 0.0)
    result = {
        "hip_y": variant.hip_y,
        "leg_scale": variant.leg_scale,
        "disk_radius": variant.disk_radius,
        "disk_com_x": com_x,
        "disk_com_z": com_z,
        **geometry,
        **score_result(forward, reverse, rest),
        "forward_distance": forward["distance"],
        "reverse_distance": reverse["distance"],
        "forward_final_speed": forward["final_speed"],
        "reverse_final_speed": reverse["final_speed"],
    }
    return result


def main(argv=None):
    args = parse_args(argv)
    combinations = list(
        itertools.product(args.hip_y, args.leg_scale, args.disk_radius, args.com_x, args.com_z)
    )
    results = []
    for index, (hip_y, leg_scale, disk_radius, com_x, com_z) in enumerate(combinations, 1):
        variant = StructureVariant(hip_y, leg_scale, disk_radius)
        result = simulate_variant(args, variant, com_x, com_z)
        results.append(result)
        print(
            f"variant={index}/{len(combinations)} hip_y={hip_y:.3f} leg_scale={leg_scale:.2f} "
            f"radius={disk_radius:.3f} com=({com_x:+.3f},{com_z:+.3f}) "
            f"distance={result['mean_coast_distance']:.3f} slip={result['mean_slip_rms']:.3f} "
            f"tilt={result['mean_axis_tilt_rms_deg']:.2f}deg rest={result['rest_drift']:+.3f} "
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

    print("\nTop rolling variants")
    for rank, result in enumerate(results[:10], 1):
        print(
            f"{rank:2d}. hip_y={result['hip_y']:.3f} leg_scale={result['leg_scale']:.2f} "
            f"radius={result['disk_radius']:.3f} com=({result['disk_com_x']:+.3f},"
            f"{result['disk_com_z']:+.3f}) score={result['score']:.4f} "
            f"distance={result['mean_coast_distance']:.3f} slip={result['mean_slip_rms']:.3f}"
        )
    print(f"saved_json={json_path.resolve()}")
    print(f"saved_csv={csv_path.resolve()}")


if __name__ == "__main__":
    main()
