from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from disk_robot.ik_reference import build_ik_reference
from disk_robot.student_policy import load_student_policy, make_student_policy_jax
from disk_robot.teacher_student_config import ForwardTeacherStudentConfig
from disk_robot_mjx.pipeline import configure_cloud_runtime, make_network_factory
from disk_robot_mjx.teacher_student_env import make_forward_teacher_student_env
from scripts.distill_forward_student import (
    _load_accepted_teacher_run,
    _reference_spec_from_teacher_run,
    _resolve_xml_path,
)


POLICIES = ("ik", "teacher", "student")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Read-only T8 straightness characterization. Roll out IK, the accepted Teacher, "
            "and T8 on paired phase-zero resets, then optionally render saved qpos locally."
        )
    )
    parser.add_argument(
        "--mode", choices=("rollout", "analyze", "render", "all"), default="rollout"
    )
    parser.add_argument("--teacher-run", type=Path)
    parser.add_argument("--student-run", type=Path)
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rollout-data", type=Path, default=None)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=130_000)
    parser.add_argument("--seed-count", type=int, default=4)
    parser.add_argument("--seed-stride", type=int, default=10_000)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--video-height", type=int, default=540)
    parser.add_argument(
        "--analysis-windows",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0, 30.0],
        help="Elapsed seconds summarized by analyze mode.",
    )
    return parser.parse_args(argv)


def _require_rollout_args(args):
    if args.teacher_run is None or args.student_run is None:
        raise SystemExit("--teacher-run and --student-run are required for rollout mode")
    if min(args.envs, args.steps, args.seed_count) < 1:
        raise SystemExit("--envs, --steps, and --seed-count must be positive")


def _config_from_t8_metadata(artifact, steps):
    metadata = artifact.metadata
    if metadata.get("stage") != "T8_PHASE_BC_NO_PREVIOUS_ACTION":
        raise SystemExit(
            "trajectory characterization requires stage T8_PHASE_BC_NO_PREVIOUS_ACTION"
        )
    stored = metadata.get("config")
    if not isinstance(stored, dict):
        raise SystemExit("T8 policy metadata has no reconstructable config")
    allowed = {field.name for field in fields(ForwardTeacherStudentConfig)}
    values = {key: value for key, value in stored.items() if key in allowed}
    config = ForwardTeacherStudentConfig(**values)
    config = replace(
        config,
        disturbance_enabled=False,
        student_phase_conditioned=True,
        student_previous_action_input=False,
        fixed_reset_phase=0.0,
        max_episode_steps=max(config.max_episode_steps, steps + 1),
    )
    if artifact.obs_mean.shape != (config.student_policy_observation_size,):
        raise SystemExit(
            "T8 normalization/config mismatch: "
            f"artifact={artifact.obs_mean.shape} config={config.student_policy_observation_size}"
        )
    if config.student_policy_observation_size != 147:
        raise SystemExit(
            f"T8 policy observation contract must be 147, got {config.student_policy_observation_size}"
        )
    if metadata.get("previous_action_input") is not False:
        raise SystemExit("T8 metadata does not confirm previous_action_input=false")
    return config


def _final_alive_indices(alive):
    alive = np.asarray(alive, dtype=bool)
    return np.maximum(np.sum(alive, axis=0).astype(np.int64) - 1, 0)


def summarize_trajectory(values, *, dt, torque_limit, ctrl_low, ctrl_high):
    xy = np.asarray(values["xy"], dtype=np.float64)
    yaw = np.asarray(values["yaw"], dtype=np.float64)
    alive = np.asarray(values["alive"], dtype=bool)
    final_index = _final_alive_indices(alive)
    env_index = np.arange(xy.shape[1])
    start_xy = xy[0]
    final_xy = xy[final_index, env_index]
    displacement = final_xy - start_xy
    dx = displacement[:, 0]
    dy = displacement[:, 1]
    path_angle = np.degrees(np.arctan2(dy, dx))
    unwrapped_yaw = np.unwrap(yaw, axis=0)
    yaw_change = unwrapped_yaw[final_index, env_index] - unwrapped_yaw[0]
    disk = np.asarray(values["disk_contact"], dtype=np.float64)
    failed = np.asarray(values["failed"], dtype=np.float64)
    action = np.asarray(values["action"], dtype=np.float64)
    force = np.asarray(values["force"], dtype=np.float64)
    ctrl = np.asarray(values["ctrl"], dtype=np.float64)
    sample_mask = alive[..., None]
    sample_count = max(int(np.sum(sample_mask)) * action.shape[-1], 1)
    ctrl_margin = np.minimum(ctrl - np.asarray(ctrl_low), np.asarray(ctrl_high) - ctrl)
    per_environment = []
    for index in range(xy.shape[1]):
        valid = alive[:, index]
        per_environment.append(
            {
                "environment": int(index),
                "alive_steps": int(np.sum(valid)),
                "duration_s": float(np.sum(valid) * dt),
                "forward_displacement_m": float(dx[index]),
                "lateral_displacement_m": float(dy[index]),
                "absolute_lateral_displacement_m": float(abs(dy[index])),
                "path_angle_deg": float(path_angle[index]),
                "yaw_change_rad": float(yaw_change[index]),
                "disk_contacts": float(np.sum(disk[:, index] * valid)),
                "failed": bool(np.max(failed[:, index]) > 0.5),
                "max_abs_action": float(np.max(np.abs(action[:, index][valid]))),
                "max_abs_force_nm": float(np.max(np.abs(force[:, index][valid]))),
            }
        )
    return {
        "environments": int(xy.shape[1]),
        "horizon_steps": int(xy.shape[0]),
        "dt": float(dt),
        "mean_forward_displacement_m": float(np.mean(dx)),
        "mean_lateral_displacement_m": float(np.mean(dy)),
        "mean_absolute_lateral_displacement_m": float(np.mean(np.abs(dy))),
        "lateral_displacement_std_m": float(np.std(dy)),
        "maximum_absolute_lateral_displacement_m": float(np.max(np.abs(dy))),
        "mean_path_angle_deg": float(np.mean(path_angle)),
        "maximum_absolute_path_angle_deg": float(np.max(np.abs(path_angle))),
        "mean_absolute_yaw_change_rad": float(np.mean(np.abs(yaw_change))),
        "maximum_absolute_yaw_change_rad": float(np.max(np.abs(yaw_change))),
        "positive_lateral_fraction": float(np.mean(dy > 0.0)),
        "failure_rate": float(np.mean(np.max(failed, axis=0) > 0.5)),
        "disk_contact_environment_rate": float(np.mean(np.sum(disk * alive, axis=0) > 0.0)),
        "disk_contacts_per_alive_step": float(np.sum(disk * alive) / max(np.sum(alive), 1)),
        "max_abs_action": float(np.max(np.abs(action[sample_mask.repeat(action.shape[-1], axis=-1)]))),
        "action_saturation_fraction": float(
            np.sum((np.abs(action) >= 0.99) & sample_mask) / sample_count
        ),
        "max_abs_force_nm": float(np.max(np.abs(force[sample_mask.repeat(force.shape[-1], axis=-1)]))),
        "force_saturation_fraction": float(
            np.sum((np.abs(force) >= 0.99 * torque_limit) & sample_mask) / sample_count
        ),
        "minimum_control_limit_margin_rad": float(
            np.min(ctrl_margin[sample_mask.repeat(ctrl.shape[-1], axis=-1)])
        ),
        "per_environment": per_environment,
    }


def characterize_gate(summaries):
    ik = summaries["ik"]
    teacher = summaries["teacher"]
    student = summaries["student"]
    student_worse = (
        student["mean_absolute_lateral_displacement_m"]
        > teacher["mean_absolute_lateral_displacement_m"] + 0.02
        or student["mean_absolute_yaw_change_rad"]
        > teacher["mean_absolute_yaw_change_rad"] + 0.15
        or student["failure_rate"] > teacher["failure_rate"] + 0.01
        or student["disk_contact_environment_rate"]
        > teacher["disk_contact_environment_rate"] + 0.01
    )
    comparable_all = all(
        summary["failure_rate"] == 0.0
        and summary["disk_contact_environment_rate"] == 0.0
        and summary["force_saturation_fraction"] < 0.01
        for summary in summaries.values()
    ) and not student_worse
    signs = [
        math.copysign(1.0, summary["mean_lateral_displacement_m"])
        if abs(summary["mean_lateral_displacement_m"]) > 0.005
        else 0.0
        for summary in (ik, teacher, student)
    ]
    repeatable_common_bias = signs[0] != 0.0 and signs[0] == signs[1] == signs[2]
    if student_worse:
        recommendation = "diagnose_t8_retention_before_t9"
    elif comparable_all:
        recommendation = "freeze_t8_and_proceed_to_t9"
    else:
        recommendation = "inspect_trajectory_artifacts_before_t9"
    return {
        "student_materially_worse_than_teacher": bool(student_worse),
        "all_policies_safe_and_student_comparable": bool(comparable_all),
        "repeatable_common_lateral_bias": bool(repeatable_common_bias),
        "recommendation": recommendation,
        "thresholds": {
            "student_extra_mean_abs_lateral_m": 0.02,
            "student_extra_mean_abs_yaw_rad": 0.15,
            "force_saturation_fraction": 0.01,
        },
    }


def analyze_time_profiles(trajectories, *, dt, windows):
    """Summarizes how paired straightness errors grow without requiring JAX/MJX."""

    horizon = min(np.asarray(trajectories[name]["xy"]).shape[0] for name in POLICIES)
    if horizon < 1 or dt <= 0.0:
        raise ValueError("saved trajectories need a positive horizon and dt")
    selected = sorted({min(max(int(round(value / dt)), 1), horizon) for value in windows})
    selected.append(horizon)
    selected = sorted(set(selected))
    profiles = {}
    per_policy_series = {}
    for name in POLICIES:
        xy = np.asarray(trajectories[name]["xy"], dtype=np.float64)[:horizon]
        yaw = np.unwrap(
            np.asarray(trajectories[name]["yaw"], dtype=np.float64)[:horizon], axis=0
        )
        origin_xy = xy[0]
        origin_yaw = yaw[0]
        dx_series = xy[..., 0] - origin_xy[:, 0]
        dy_series = xy[..., 1] - origin_xy[:, 1]
        yaw_series = yaw - origin_yaw
        per_policy_series[name] = (dx_series, dy_series, yaw_series)
        entries = []
        for count in selected:
            index = count - 1
            dx = dx_series[index]
            dy = dy_series[index]
            yaw_change = yaw_series[index]
            entries.append(
                {
                    "steps": int(count),
                    "elapsed_s": float(count * dt),
                    "mean_forward_displacement_m": float(np.mean(dx)),
                    "mean_lateral_displacement_m": float(np.mean(dy)),
                    "mean_absolute_lateral_displacement_m": float(np.mean(np.abs(dy))),
                    "lateral_displacement_std_m": float(np.std(dy)),
                    "positive_lateral_fraction": float(np.mean(dy > 0.0)),
                    "mean_absolute_yaw_change_rad": float(np.mean(np.abs(yaw_change))),
                    "mean_drift_ratio": float(
                        np.mean(np.abs(dy) / np.maximum(np.abs(dx), 1e-6))
                    ),
                }
            )
        profiles[name] = entries

    paired = []
    for position, count in enumerate(selected):
        index = count - 1
        teacher_dx, teacher_dy, teacher_yaw = per_policy_series["teacher"]
        student_dx, student_dy, student_yaw = per_policy_series["student"]
        paired.append(
            {
                "steps": int(count),
                "elapsed_s": float(count * dt),
                "student_minus_teacher_forward_m": float(
                    np.mean(student_dx[index] - teacher_dx[index])
                ),
                "student_minus_teacher_signed_lateral_m": float(
                    np.mean(student_dy[index] - teacher_dy[index])
                ),
                "student_minus_teacher_absolute_lateral_m": float(
                    np.mean(np.abs(student_dy[index]) - np.abs(teacher_dy[index]))
                ),
                "student_minus_teacher_absolute_yaw_rad": float(
                    np.mean(np.abs(student_yaw[index]) - np.abs(teacher_yaw[index]))
                ),
                "student_more_lateral_fraction": float(
                    np.mean(np.abs(student_dy[index]) > np.abs(teacher_dy[index]))
                ),
            }
        )

    student_abs = np.mean(np.abs(per_policy_series["student"][1]), axis=1)
    teacher_abs = np.mean(np.abs(per_policy_series["teacher"][1]), axis=1)
    excess = student_abs - teacher_abs
    candidates = np.flatnonzero(excess > 0.02)
    onset_step = int(candidates[0] + 1) if candidates.size else None
    return {
        "dt": float(dt),
        "horizon_steps": int(horizon),
        "duration_s": float(horizon * dt),
        "profiles": profiles,
        "paired_student_minus_teacher": paired,
        "student_excess_abs_lateral_2cm_onset_step": onset_step,
        "student_excess_abs_lateral_2cm_onset_s": (
            float(onset_step * dt) if onset_step is not None else None
        ),
    }


def analyze_saved_rollout(path, out=None, windows=(5.0, 10.0, 20.0, 30.0)):
    path = path.expanduser().resolve()
    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        trajectories = {
            name: {
                "xy": archive[f"{name}_xy"].copy(),
                "yaw": archive[f"{name}_yaw"].copy(),
            }
            for name in POLICIES
        }
    report = analyze_time_profiles(
        trajectories, dt=float(metadata["dt"]), windows=windows
    )
    output_dir = out.expanduser().resolve() if out is not None else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "trajectory_time_analysis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for entry in report["paired_student_minus_teacher"]:
        print(
            f"stage=t8_time_analysis elapsed_s={entry['elapsed_s']:.1f} "
            f"delta_forward={entry['student_minus_teacher_forward_m']:+.4f} "
            f"delta_lateral_abs={entry['student_minus_teacher_absolute_lateral_m']:+.4f} "
            f"delta_yaw_abs={entry['student_minus_teacher_absolute_yaw_rad']:+.4f} "
            f"student_more_lateral={entry['student_more_lateral_fraction']:.3f}",
            flush=True,
        )
    print(
        "stage=t8_time_analysis_result "
        f"excess_2cm_onset_s={report['student_excess_abs_lateral_2cm_onset_s']} "
        f"report={report_path}",
        flush=True,
    )
    return report_path


def _rollout_policy(jax, jp, env, kind, teacher_policy, student_policy, seed, envs, steps):
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step))
    actuator_ids = env.actuator_ids
    torso_body_id = env.torso_body_id
    state = reset_batch(jax.random.split(jax.random.PRNGKey(seed), envs))

    def scan_step(carry, _):
        current, policy_key = carry
        policy_key, action_key = jax.random.split(policy_key)
        if kind == "ik":
            action = jp.zeros((envs, env.action_size))
        elif kind == "teacher":
            action, _ = teacher_policy(current.obs, action_key)
        else:
            action = student_policy(current.obs)
        next_state = step_batch(current, action)
        mat = next_state.pipeline_state.xmat[:, torso_body_id]
        yaw = jp.arctan2(mat[:, 1, 0], mat[:, 0, 0])
        alive = 1.0 - current.done
        return (next_state, policy_key), {
            "xy": next_state.pipeline_state.xpos[:, torso_body_id, :2],
            "yaw": yaw,
            "disk_contact": next_state.metrics["disk_contact_count"],
            "failed": next_state.metrics["failed"],
            "alive": alive,
            "action": next_state.info["student_action"],
            "force": next_state.pipeline_state.actuator_force[:, actuator_ids],
            "ctrl": next_state.pipeline_state.ctrl[:, actuator_ids],
            "qpos": next_state.pipeline_state.qpos[0],
        }

    (_, _), values = jax.lax.scan(
        scan_step, (state, jax.random.PRNGKey(seed + 1)), (), length=steps
    )
    return {key: np.asarray(jax.device_get(value)) for key, value in values.items()}


def _write_xy_svg(path, trajectories):
    colors = {"ik": "#2878b5", "teacher": "#e07a1f", "student": "#2a9d55"}
    points = np.concatenate([value.reshape(-1, 2) for value in trajectories.values()])
    xmin, ymin = np.min(points, axis=0)
    xmax, ymax = np.max(points, axis=0)
    xpad = max((xmax - xmin) * 0.08, 0.02)
    ypad = max((ymax - ymin) * 0.12, 0.02)
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad
    width, height, margin = 1000, 620, 60

    def project(values):
        x = margin + (values[:, 0] - xmin) / max(xmax - xmin, 1e-9) * (width - 2 * margin)
        y = height - margin - (values[:, 1] - ymin) / max(ymax - ymin, 1e-9) * (height - 2 * margin)
        return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="60" y="30" font-family="sans-serif" font-size="20">Paired phase-zero XY trajectories (one rollout each)</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#aaa"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#aaa"/>',
    ]
    for index, (name, values) in enumerate(trajectories.items()):
        lines.append(
            f'<polyline points="{project(values)}" fill="none" stroke="{colors[name]}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="{width-240}" y="{50 + index*26}" font-family="sans-serif" font-size="16" fill="{colors[name]}">{name}</text>'
        )
    lines.extend(
        (
            f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="sans-serif">world x (m), range {xmin:.3f} to {xmax:.3f}</text>',
            f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif">world y (m), range {ymin:.3f} to {ymax:.3f}</text>',
            "</svg>",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_rollout(args):
    _require_rollout_args(args)
    teacher_run, teacher_run_config, teacher_evaluation, _, params_path = (
        _load_accepted_teacher_run(args.teacher_run)
    )
    student_run = args.student_run.expanduser().resolve()
    student_path = student_run / "student_policy_phase_bc_no_previous_action.npz"
    if not student_path.exists():
        raise SystemExit(f"T8 Student policy is missing: {student_path}")
    artifact = load_student_policy(student_path)
    config = _config_from_t8_metadata(artifact, args.steps)
    reference_spec = _reference_spec_from_teacher_run(teacher_run_config)
    config = replace(config, student_phase_frequency=reference_spec.frequency)
    xml_path = _resolve_xml_path(teacher_run_config, args.xml_path)
    if not xml_path.exists():
        raise SystemExit(f"robot XML does not exist: {xml_path}")
    if args.out is None:
        args.out = student_run / "t8_trajectory_characterization"
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    configure_cloud_runtime(mujoco_gl=args.mujoco_gl, verbose=True)
    try:
        import jax
        import jax.numpy as jp
        from brax.io import model as model_io
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise SystemExit(f"T8 rollout requires the mjx312 stack: {exc}") from exc

    reference = build_ik_reference(xml_path, reference_spec)
    teacher_env = make_forward_teacher_student_env(
        "teacher", config=config, reference=reference, xml_path=xml_path, seed=0
    )
    ik_env = make_forward_teacher_student_env(
        "teacher", config=config, reference=reference, xml_path=xml_path, seed=0
    )
    student_env = make_forward_teacher_student_env(
        "student", config=config, xml_path=xml_path, seed=0
    )
    networks = make_network_factory(
        teacher_run_config.get("teacher_hidden", [256, 256, 128]), "elu"
    )(
        observation_size=teacher_env.observation_size,
        action_size=teacher_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    teacher_policy = ppo_networks.make_inference_fn(networks)(
        model_io.load_params(params_path), deterministic=True
    )
    student_policy = jax.jit(make_student_policy_jax(artifact))
    env_by_policy = {"ik": ik_env, "teacher": teacher_env, "student": student_env}
    accumulated = {name: [] for name in POLICIES}
    qpos_single = {}
    seeds = [args.seed + index * args.seed_stride for index in range(args.seed_count)]
    for seed_index, seed in enumerate(seeds):
        for name in POLICIES:
            values = _rollout_policy(
                jax,
                jp,
                env_by_policy[name],
                name,
                teacher_policy,
                student_policy,
                seed,
                args.envs,
                args.steps,
            )
            if seed_index == 0:
                qpos_single[name] = values.pop("qpos")
            else:
                values.pop("qpos")
            accumulated[name].append(values)
        print(
            f"stage=t8_characterization_batch batch={seed_index + 1}/{len(seeds)} seed={seed}",
            flush=True,
        )

    merged = {}
    summaries = {}
    for name in POLICIES:
        merged[name] = {
            key: np.concatenate([batch[key] for batch in accumulated[name]], axis=1)
            for key in accumulated[name][0]
        }
        summaries[name] = summarize_trajectory(
            merged[name],
            dt=env_by_policy[name].dt,
            torque_limit=config.torque_limit,
            ctrl_low=np.asarray(env_by_policy[name].contract.ctrl_low),
            ctrl_high=np.asarray(env_by_policy[name].contract.ctrl_high),
        )
    gate = characterize_gate(summaries)
    report = {
        "stage": "T8_TRAJECTORY_CHARACTERIZATION",
        "read_only": True,
        "teacher_run": str(teacher_run),
        "teacher_selected_step": int(teacher_evaluation["selected_step"]),
        "student_policy": str(student_path),
        "xml_path": str(xml_path),
        "paired_fixed_reset_phase": 0.0,
        "seeds": seeds,
        "environments_per_seed": args.envs,
        "steps": args.steps,
        "duration_s": args.steps * student_env.dt,
        "summaries": summaries,
        "decision_gate": gate,
    }
    report_path = out / "trajectory_characterization.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    npz_values = {
        "metadata_json": np.asarray(json.dumps({"xml_path": str(xml_path), "dt": student_env.dt})),
    }
    for name in POLICIES:
        npz_values[f"{name}_qpos"] = qpos_single[name]
        for key in ("xy", "yaw", "alive", "disk_contact", "failed"):
            npz_values[f"{name}_{key}"] = merged[name][key]
    rollout_path = out / "trajectory_rollouts.npz"
    np.savez_compressed(rollout_path, **npz_values)
    _write_xy_svg(out / "xy_trajectories.svg", {name: merged[name]["xy"][:, 0] for name in POLICIES})
    for name in POLICIES:
        summary = summaries[name]
        print(
            f"stage=t8_trajectory policy={name} forward={summary['mean_forward_displacement_m']:.4f} "
            f"lateral_abs={summary['mean_absolute_lateral_displacement_m']:.4f} "
            f"lateral_std={summary['lateral_displacement_std_m']:.4f} "
            f"yaw_abs={summary['mean_absolute_yaw_change_rad']:.4f} "
            f"failure={summary['failure_rate']:.3f} disk={summary['disk_contact_environment_rate']:.3f} "
            f"force_sat={summary['force_saturation_fraction']:.5f}",
            flush=True,
        )
    print(
        f"stage=t8_characterization_gate recommendation={gate['recommendation']} "
        f"report={report_path} rollout={rollout_path}",
        flush=True,
    )
    return rollout_path


def render_rollout(path, out, fps=30, width=960, height=540, xml_override=None):
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(f"local rendering requires MuJoCo: {exc}") from exc
    from disk_robot.model_contract import resolve_model_contract
    from disk_robot.video_recorder import MujocoVideoRecorder

    path = path.expanduser().resolve()
    with np.load(path) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        qpos = {name: archive[f"{name}_qpos"].copy() for name in POLICIES}
    xml_path = (
        xml_override.expanduser().resolve()
        if xml_override is not None
        else Path(metadata["xml_path"])
    )
    if not xml_path.exists():
        raise SystemExit(f"saved XML path is unavailable locally; pass/copy the original tree: {xml_path}")
    out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name in POLICIES:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        contract = resolve_model_contract(model)
        video_path = out / f"{name}_tracking_side.mp4"
        with MujocoVideoRecorder(
            model,
            video_path,
            contract.torso_body_id,
            fps=fps,
            width=width,
            height=height,
            azimuth=90.0,
            elevation=-12.0,
            distance=1.4,
        ) as recorder:
            for step, pose in enumerate(qpos[name]):
                data.qpos[:] = pose
                data.time = step * float(metadata["dt"])
                mujoco.mj_forward(model, data)
                recorder.capture(data)
        outputs.append(str(video_path))
        print(f"stage=t8_render policy={name} video={video_path}", flush=True)
    return outputs


def main(argv=None):
    args = parse_args(argv)
    rollout_path = None
    if args.mode in ("rollout", "all"):
        rollout_path = run_rollout(args)
    if args.mode == "analyze":
        if args.rollout_data is None:
            raise SystemExit("--rollout-data is required for analyze mode")
        analyze_saved_rollout(args.rollout_data, args.out, args.analysis_windows)
    if args.mode in ("render", "all"):
        selected = rollout_path or args.rollout_data
        if selected is None:
            raise SystemExit("--rollout-data is required for render mode")
        out = (
            args.out.expanduser().resolve()
            if args.out is not None
            else selected.expanduser().resolve().parent
        )
        render_rollout(
            selected,
            out,
            args.fps,
            args.video_width,
            args.video_height,
            args.xml_path,
        )


if __name__ == "__main__":
    main()
