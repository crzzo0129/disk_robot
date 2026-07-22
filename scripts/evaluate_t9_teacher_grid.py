from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from disk_robot.ik_reference import IKReferenceSpec, build_ik_reference_bank
from disk_robot.t9_command import make_t9_config, validate_forward_speed_anchors
from disk_robot_mjx.pipeline import configure_cloud_runtime, make_network_factory
from disk_robot_mjx.teacher_student_env import make_forward_teacher_student_env
from scripts.characterize_t8_trajectories import _rollout_policy, summarize_trajectory
from scripts.distill_forward_student import _config_from_teacher_run, _resolve_xml_path
from scripts.train_forward_teacher_student import (
    _disturbed_teacher_score,
    _evaluate_teacher,
    _print_evaluation_summary,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate a preliminary T9 Teacher independently at every vx anchor."
    )
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--long-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--long-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=220_000)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read T9 artifact {path}: {exc}") from exc


def _speed_gate(
    speed,
    ppo_nominal,
    baseline_nominal,
    ppo_disturbed,
    baseline_disturbed,
    long_ppo,
    long_baseline,
):
    velocity_tolerance = 0.015 if speed > 0.0 else 0.02
    nominal_preserved = (
        ppo_nominal["failure_rate"] <= baseline_nominal["failure_rate"] + 0.02
        and ppo_nominal["mean_velocity_error"]
        <= baseline_nominal["mean_velocity_error"] + velocity_tolerance
        and ppo_nominal["mean_roll_pitch_rate_rms"]
        <= baseline_nominal["mean_roll_pitch_rate_rms"] + 0.10
        and ppo_nominal["mean_abs_velocity_y"]
        <= baseline_nominal["mean_abs_velocity_y"] + 0.015
        and ppo_nominal["mean_abs_yaw_rate"]
        <= baseline_nominal["mean_abs_yaw_rate"] + 0.06
    )
    disturbed_improved = (
        ppo_disturbed["failure_rate"] <= baseline_disturbed["failure_rate"]
        and ppo_disturbed["mean_post_push_velocity_error"]
        <= baseline_disturbed["mean_post_push_velocity_error"] + 0.01
        and ppo_disturbed["mean_recovery_time"]
        <= baseline_disturbed["mean_recovery_time"] + 0.10
        and _disturbed_teacher_score(ppo_disturbed)
        >= _disturbed_teacher_score(baseline_disturbed) - 0.01
    )
    long_safe = (
        long_ppo["failure_rate"] == 0.0
        and long_ppo["disk_contact_environment_rate"] == 0.0
        and long_ppo["force_saturation_fraction"] < 0.01
        and long_ppo["mean_absolute_lateral_displacement_m"]
        <= long_baseline["mean_absolute_lateral_displacement_m"] + 0.25
        and long_ppo["mean_absolute_yaw_change_rad"]
        <= long_baseline["mean_absolute_yaw_change_rad"] + 0.20
    )
    return {
        "accepted": bool(nominal_preserved and disturbed_improved and long_safe),
        "nominal_preserved": bool(nominal_preserved),
        "disturbed_preserved_or_improved": bool(disturbed_improved),
        "long_horizon_safe": bool(long_safe),
    }


def main(argv=None):
    args = parse_args(argv)
    if min(args.eval_envs, args.long_envs, args.steps, args.long_steps) < 1:
        raise SystemExit("evaluation counts must be positive")
    teacher_run = args.teacher_run.expanduser().resolve()
    run_config = _read_json(teacher_run / "run_config.json")
    evaluation_path = teacher_run / "teacher" / "evaluation.json"
    evaluation = _read_json(evaluation_path)
    selection = _read_json(teacher_run / "teacher" / "selection.json")
    if run_config.get("stage") != "T9_FORWARD_COMMAND_TEACHER":
        raise SystemExit("teacher run is not a T9 command-grid run")
    if selection.get("selected_source") != "ppo":
        raise SystemExit("T9 grid validation requires a selected PPO Teacher")
    params_path = teacher_run / "teacher" / "params"
    if not params_path.exists():
        raise SystemExit(f"T9 Teacher params are missing: {params_path}")
    anchors = validate_forward_speed_anchors(run_config.get("command_vx_grid", ()))
    specs = tuple(
        IKReferenceSpec(**values)
        for values in run_config.get("resolved_ik_reference_bank", ())
    )
    if len(specs) != len(anchors):
        raise SystemExit("T9 run_config reference bank does not match its command grid")
    xml_path = _resolve_xml_path(run_config, args.xml_path)
    base_config = make_t9_config(_config_from_teacher_run(run_config), anchors)
    reference = build_ik_reference_bank(xml_path, anchors, specs)

    configure_cloud_runtime(mujoco_gl=args.mujoco_gl, verbose=True)
    try:
        import jax
        import jax.numpy as jp
        from brax.io import model as model_io
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise SystemExit(f"T9 grid evaluation requires the mjx312 stack: {exc}") from exc

    template_env = make_forward_teacher_student_env(
        "teacher", config=base_config, reference=reference, xml_path=xml_path, seed=args.seed
    )
    networks = make_network_factory(
        run_config.get("teacher_hidden", [256, 256, 128]), "elu"
    )(
        observation_size=template_env.observation_size,
        action_size=template_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    teacher_policy = ppo_networks.make_inference_fn(networks)(
        model_io.load_params(params_path), deterministic=True
    )
    zero_policy = lambda obs, rng: (
        jp.zeros(obs.shape[:-1] + (template_env.action_size,)),
        {},
    )
    reports = {}
    for index, speed in enumerate(anchors):
        fixed = replace(
            base_config,
            command_vx=speed,
            command_vx_values=(),
            disturbance_enabled=False,
            max_episode_steps=max(args.long_steps + 1, base_config.max_episode_steps),
        )
        disturbed = replace(fixed, disturbance_enabled=True)
        nominal_env = make_forward_teacher_student_env(
            "teacher", config=fixed, reference=reference, xml_path=xml_path, seed=args.seed + index
        )
        disturbed_env = make_forward_teacher_student_env(
            "teacher", config=disturbed, reference=reference, xml_path=xml_path, seed=args.seed + index
        )
        ppo_nominal = _evaluate_teacher(
            jax, nominal_env, teacher_policy, args.seed + 1000 * index, args.eval_envs, args.steps
        )
        ik_nominal = _evaluate_teacher(
            jax, nominal_env, zero_policy, args.seed + 1000 * index, args.eval_envs, args.steps
        )
        ppo_disturbed = _evaluate_teacher(
            jax, disturbed_env, teacher_policy, args.seed + 500 + 1000 * index, args.eval_envs, args.steps
        )
        ik_disturbed = _evaluate_teacher(
            jax, disturbed_env, zero_policy, args.seed + 500 + 1000 * index, args.eval_envs, args.steps
        )
        ppo_trace = _rollout_policy(
            jax, jp, nominal_env, "teacher", teacher_policy, None,
            args.seed + 700 + 1000 * index, args.long_envs, args.long_steps,
        )
        ik_trace = _rollout_policy(
            jax, jp, nominal_env, "ik", teacher_policy, None,
            args.seed + 700 + 1000 * index, args.long_envs, args.long_steps,
        )
        ppo_trace.pop("qpos")
        ik_trace.pop("qpos")
        long_ppo = summarize_trajectory(
            ppo_trace, dt=nominal_env.dt, torque_limit=fixed.torque_limit,
            ctrl_low=nominal_env.contract.ctrl_low, ctrl_high=nominal_env.contract.ctrl_high,
        )
        long_ik = summarize_trajectory(
            ik_trace, dt=nominal_env.dt, torque_limit=fixed.torque_limit,
            ctrl_low=nominal_env.contract.ctrl_low, ctrl_high=nominal_env.contract.ctrl_high,
        )
        gate = _speed_gate(
            speed,
            ppo_nominal,
            ik_nominal,
            ppo_disturbed,
            ik_disturbed,
            long_ppo,
            long_ik,
        )
        reports[f"{speed:.2f}"] = {
            **gate,
            "command_vx": speed,
            "ppo_nominal": ppo_nominal,
            "ik_nominal": ik_nominal,
            "ppo_disturbed": ppo_disturbed,
            "ik_disturbed": ik_disturbed,
            "ppo_long_horizon": long_ppo,
            "ik_long_horizon": long_ik,
        }
        _print_evaluation_summary("t9_teacher_grid", ppo_nominal, f"vx_{speed:.2f}")
        print(
            f"stage=t9_teacher_speed_gate vx={speed:.2f} accepted={gate['accepted']} "
            f"nominal={gate['nominal_preserved']} disturbed={gate['disturbed_preserved_or_improved']} "
            f"long={gate['long_horizon_safe']} lateral30={long_ppo['mean_absolute_lateral_displacement_m']:.4f}",
            flush=True,
        )

    accepted = all(report["accepted"] for report in reports.values())
    grid_report = {
        "stage": "T9_FORWARD_COMMAND_TEACHER_GRID_EVALUATION",
        "accepted": accepted,
        "anchors": list(anchors),
        "steps": args.steps,
        "long_steps": args.long_steps,
        "speed_reports": reports,
    }
    grid_path = teacher_run / "teacher" / "grid_evaluation.json"
    grid_path.write_text(json.dumps(grid_report, indent=2), encoding="utf-8")
    evaluation["accepted"] = accepted
    evaluation["grid_validation_pending"] = False
    evaluation["grid_evaluation"] = str(grid_path)
    evaluation["stage"] = "T9_FORWARD_COMMAND_TEACHER"
    if accepted:
        evaluation.pop("rejection_reason", None)
    else:
        evaluation["rejection_reason"] = "t9_speed_grid_gate_failed"
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(
        f"stage=t9_teacher_grid_acceptance accepted={accepted} report={grid_path}",
        flush=True,
    )
    if args.strict and not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
