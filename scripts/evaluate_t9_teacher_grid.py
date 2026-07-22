from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=None,
        help="Evaluate only these stored command anchors (diagnostic runs are not promoted).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed speed gate, after saving its partial report.",
    )
    parser.add_argument(
        "--disturbed-only",
        action="store_true",
        help="Diagnostic mode: run only PPO/IK disturbed checks; requires --speeds.",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Diagnostic mode: run only PPO/IK long-horizon checks; requires --speeds.",
    )
    parser.add_argument(
        "--residual-scale-multiplier",
        type=float,
        default=None,
        help="Diagnostic override relative to the base residual limits.",
    )
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read T9 artifact {path}: {exc}") from exc


def _candidate_params_path(teacher_run, selection):
    """Returns the PPO candidate that the T9 grid gate must independently judge."""
    teacher_dir = teacher_run / "teacher"
    candidates = []
    if selection.get("selected_source") == "ppo":
        candidates.append(("selected_ppo", teacher_dir / "params"))
    candidates.extend(
        (
            ("ppo_best", teacher_dir / "params_ppo_best"),
            ("ppo_final", teacher_dir / "params_final"),
        )
    )
    for source, path in candidates:
        if path.exists():
            return source, path
    raise SystemExit(
        "T9 grid validation requires a saved PPO candidate "
        "(teacher/params_ppo_best or teacher/params_final)"
    )


def _selected_speeds(anchors, requested):
    if requested is None:
        return tuple(anchors)
    selected = []
    for raw_value in requested:
        matches = [value for value in anchors if abs(value - float(raw_value)) < 1e-9]
        if not matches:
            choices = ", ".join(f"{value:g}" for value in anchors)
            raise SystemExit(
                f"requested diagnostic speed {raw_value:g} is not a stored anchor; "
                f"choose from {choices}"
            )
        if matches[0] in selected:
            raise SystemExit(f"duplicate diagnostic speed: {raw_value:g}")
        selected.append(matches[0])
    return tuple(selected)


def _run_timed(speed, name, operation):
    print(f"stage=t9_teacher_eval_begin vx={speed:.2f} check={name}", flush=True)
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    print(
        f"stage=t9_teacher_eval_done vx={speed:.2f} check={name} "
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return result


def _disturbed_gate_checks(speed, ppo_disturbed, baseline_disturbed):
    checks = {
        "failure_rate": ppo_disturbed["failure_rate"]
        <= baseline_disturbed["failure_rate"],
        "post_push_velocity_error": ppo_disturbed["mean_post_push_velocity_error"]
        <= baseline_disturbed["mean_post_push_velocity_error"] + 0.01,
        "recovery_time": ppo_disturbed["mean_recovery_time"]
        <= baseline_disturbed["mean_recovery_time"] + 0.10,
    }
    if speed > 0.0:
        checks["disturbed_score"] = _disturbed_teacher_score(
            ppo_disturbed
        ) >= _disturbed_teacher_score(baseline_disturbed) - 0.01
    else:
        # At stop, the generic score favors exact zero-residual IK and includes
        # forward distance.  Gate the physical disturbed behavior directly.
        checks.update(
            post_push_lateral_speed=ppo_disturbed["mean_post_push_abs_velocity_y"]
            <= baseline_disturbed["mean_post_push_abs_velocity_y"] + 0.02,
            yaw_rate=ppo_disturbed["mean_abs_yaw_rate"]
            <= baseline_disturbed["mean_abs_yaw_rate"] + 0.06,
            disk_contact=ppo_disturbed["mean_disk_contacts"]
            <= baseline_disturbed["mean_disk_contacts"] + 0.02,
        )
    return checks


def _print_disturbed_gate(speed, checks, ppo_disturbed, ik_disturbed):
    score_passed = _disturbed_teacher_score(ppo_disturbed) >= (
        _disturbed_teacher_score(ik_disturbed) - 0.01
    )
    stop_detail = ""
    if speed == 0.0:
        stop_detail = (
            f" post_lateral={checks['post_push_lateral_speed']}"
            f" ppo_post_vy={ppo_disturbed['mean_post_push_abs_velocity_y']:.4f}"
            f" ik_post_vy={ik_disturbed['mean_post_push_abs_velocity_y']:.4f}"
            f" yaw={checks['yaw_rate']} disk={checks['disk_contact']}"
        )
    print(
        f"stage=t9_teacher_disturbed_gate vx={speed:.2f} "
        f"accepted={all(checks.values())} "
        f"failure={checks['failure_rate']} "
        f"ppo_failure={ppo_disturbed['failure_rate']:.3f} "
        f"ik_failure={ik_disturbed['failure_rate']:.3f} "
        f"post_error={checks['post_push_velocity_error']} "
        f"ppo_post_error={ppo_disturbed['mean_post_push_velocity_error']:.4f} "
        f"ik_post_error={ik_disturbed['mean_post_push_velocity_error']:.4f} "
        f"recovery={checks['recovery_time']} "
        f"ppo_recovery_s={ppo_disturbed['mean_recovery_time']:.3f} "
        f"ik_recovery_s={ik_disturbed['mean_recovery_time']:.3f} "
        f"score={score_passed} score_required={speed > 0.0} "
        f"ppo_score={_disturbed_teacher_score(ppo_disturbed):.4f} "
        f"ik_score={_disturbed_teacher_score(ik_disturbed):.4f}"
        f"{stop_detail}",
        flush=True,
    )


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
    nominal_checks = {
        "failure_rate": ppo_nominal["failure_rate"]
        <= baseline_nominal["failure_rate"] + 0.02,
        "velocity_error": ppo_nominal["mean_velocity_error"]
        <= baseline_nominal["mean_velocity_error"] + velocity_tolerance,
        "roll_pitch_rate": ppo_nominal["mean_roll_pitch_rate_rms"]
        <= baseline_nominal["mean_roll_pitch_rate_rms"] + 0.10,
        "lateral_speed": ppo_nominal["mean_abs_velocity_y"]
        <= baseline_nominal["mean_abs_velocity_y"] + 0.015,
        "yaw_rate": ppo_nominal["mean_abs_yaw_rate"]
        <= baseline_nominal["mean_abs_yaw_rate"] + 0.06,
    }
    disturbed_checks = _disturbed_gate_checks(
        speed, ppo_disturbed, baseline_disturbed
    )
    long_checks = _long_gate_checks(long_ppo, long_baseline)

    nominal_preserved = all(nominal_checks.values())
    disturbed_improved = all(disturbed_checks.values())
    long_safe = all(long_checks.values())
    return {
        "accepted": bool(nominal_preserved and disturbed_improved and long_safe),
        "nominal_preserved": bool(nominal_preserved),
        "disturbed_preserved_or_improved": bool(disturbed_improved),
        "long_horizon_safe": bool(long_safe),
        "checks": {
            "nominal": nominal_checks,
            "disturbed": disturbed_checks,
            "long_horizon": long_checks,
        },
    }


def _long_gate_checks(long_ppo, long_baseline):
    return {
        "failure_rate": long_ppo["failure_rate"] == 0.0,
        "disk_contact": long_ppo["disk_contact_environment_rate"] == 0.0,
        "force_saturation": long_ppo["force_saturation_fraction"] < 0.01,
        "lateral_displacement": long_ppo["mean_absolute_lateral_displacement_m"]
        <= long_baseline["mean_absolute_lateral_displacement_m"] + 0.25,
        "yaw_change": long_ppo["mean_absolute_yaw_change_rad"]
        <= long_baseline["mean_absolute_yaw_change_rad"] + 0.20,
    }


def main(argv=None):
    args = parse_args(argv)
    if min(args.eval_envs, args.long_envs, args.steps, args.long_steps) < 1:
        raise SystemExit("evaluation counts must be positive")
    if args.disturbed_only and args.long_only:
        raise SystemExit("choose only one of --disturbed-only and --long-only")
    if (args.disturbed_only or args.long_only) and args.speeds is None:
        raise SystemExit("diagnostic-only modes require an explicit --speeds selection")
    if args.residual_scale_multiplier is not None:
        if not (args.disturbed_only or args.long_only):
            raise SystemExit("residual-scale override is diagnostic-only")
        if args.residual_scale_multiplier <= 0.0:
            raise SystemExit("--residual-scale-multiplier must be positive")
    teacher_run = args.teacher_run.expanduser().resolve()
    run_config = _read_json(teacher_run / "run_config.json")
    evaluation_path = teacher_run / "teacher" / "evaluation.json"
    evaluation = _read_json(evaluation_path)
    selection = _read_json(teacher_run / "teacher" / "selection.json")
    if run_config.get("stage") != "T9_FORWARD_COMMAND_TEACHER":
        raise SystemExit("teacher run is not a T9 command-grid run")
    candidate_source, candidate_params_path = _candidate_params_path(
        teacher_run, selection
    )
    candidate_step = int(selection.get("ppo_step", selection.get("selected_step", 0)))
    print(
        f"stage=t9_teacher_candidate source={candidate_source} "
        f"step={candidate_step:,} params={candidate_params_path}",
        flush=True,
    )
    anchors = validate_forward_speed_anchors(run_config.get("command_vx_grid", ()))
    selected_speeds = _selected_speeds(anchors, args.speeds)
    print(
        "stage=t9_teacher_grid_plan "
        f"stored={','.join(f'{value:g}' for value in anchors)} "
        f"selected={','.join(f'{value:g}' for value in selected_speeds)} "
        f"eval_envs={args.eval_envs} steps={args.steps} "
        f"long_envs={args.long_envs} long_steps={args.long_steps} "
        f"fail_fast={args.fail_fast}",
        flush=True,
    )
    specs = tuple(
        IKReferenceSpec(**values)
        for values in run_config.get("resolved_ik_reference_bank", ())
    )
    if len(specs) != len(anchors):
        raise SystemExit("T9 run_config reference bank does not match its command grid")
    xml_path = _resolve_xml_path(run_config, args.xml_path)
    base_config = make_t9_config(_config_from_teacher_run(run_config), anchors)
    if args.residual_scale_multiplier is not None:
        stored_multiplier = float(run_config.get("residual_scale_multiplier", 1.0))
        ratio = args.residual_scale_multiplier / stored_multiplier
        base_config = replace(
            base_config,
            residual_scale=tuple(value * ratio for value in base_config.residual_scale),
        )
        print(
            "stage=t9_teacher_residual_scale_override "
            f"stored={stored_multiplier:g} diagnostic={args.residual_scale_multiplier:g}",
            flush=True,
        )
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
    candidate_params = model_io.load_params(candidate_params_path)
    teacher_policy = ppo_networks.make_inference_fn(networks)(
        candidate_params, deterministic=True
    )
    zero_policy = lambda obs, rng: (
        jp.zeros(obs.shape[:-1] + (template_env.action_size,)),
        {},
    )
    reports = {}
    partial_path = teacher_run / "teacher" / "grid_evaluation.partial.json"
    disturbed_diagnosis_path = (
        teacher_run / "teacher" / "grid_disturbed_diagnosis.json"
    )
    long_diagnosis_path = teacher_run / "teacher" / "grid_long_diagnosis.json"
    for speed in selected_speeds:
        index = anchors.index(speed)
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
        nominal_reset_batch = jax.jit(jax.vmap(nominal_env.reset))
        nominal_step_batch = jax.jit(jax.vmap(nominal_env.step))
        disturbed_reset_batch = jax.jit(jax.vmap(disturbed_env.reset))
        disturbed_step_batch = jax.jit(jax.vmap(disturbed_env.step))
        common_nominal = {
            "reset_batch": nominal_reset_batch,
            "step_batch": nominal_step_batch,
        }
        common_disturbed = {
            "reset_batch": disturbed_reset_batch,
            "step_batch": disturbed_step_batch,
        }
        if args.long_only:
            ppo_trace = _run_timed(
                speed,
                "ppo_long",
                lambda: _rollout_policy(
                    jax,
                    jp,
                    nominal_env,
                    "teacher",
                    teacher_policy,
                    None,
                    args.seed + 700 + 1000 * index,
                    args.long_envs,
                    args.long_steps,
                    **common_nominal,
                ),
            )
            ik_trace = _run_timed(
                speed,
                "ik_long",
                lambda: _rollout_policy(
                    jax,
                    jp,
                    nominal_env,
                    "ik",
                    teacher_policy,
                    None,
                    args.seed + 700 + 1000 * index,
                    args.long_envs,
                    args.long_steps,
                    **common_nominal,
                ),
            )
            ppo_trace.pop("qpos")
            ik_trace.pop("qpos")
            long_ppo = summarize_trajectory(
                ppo_trace,
                dt=nominal_env.dt,
                torque_limit=fixed.torque_limit,
                ctrl_low=nominal_env.contract.ctrl_low,
                ctrl_high=nominal_env.contract.ctrl_high,
            )
            long_ik = summarize_trajectory(
                ik_trace,
                dt=nominal_env.dt,
                torque_limit=fixed.torque_limit,
                ctrl_low=nominal_env.contract.ctrl_low,
                ctrl_high=nominal_env.contract.ctrl_high,
            )
            long_checks = _long_gate_checks(long_ppo, long_ik)
            reports[f"{speed:.2f}"] = {
                "accepted": bool(all(long_checks.values())),
                "command_vx": speed,
                "checks": long_checks,
                "ppo_long_horizon": long_ppo,
                "ik_long_horizon": long_ik,
            }
            print(
                f"stage=t9_teacher_long_diagnosis vx={speed:.2f} "
                f"accepted={all(long_checks.values())} "
                f"ppo_lateral={long_ppo['mean_absolute_lateral_displacement_m']:.4f} "
                f"ik_lateral={long_ik['mean_absolute_lateral_displacement_m']:.4f} "
                f"lateral={long_checks['lateral_displacement']} "
                f"ppo_yaw={long_ppo['mean_absolute_yaw_change_rad']:.4f} "
                f"ik_yaw={long_ik['mean_absolute_yaw_change_rad']:.4f} "
                f"yaw={long_checks['yaw_change']}",
                flush=True,
            )
            long_diagnosis_path.write_text(
                json.dumps(
                    {
                        "stage": "T9_FORWARD_COMMAND_TEACHER_LONG_DIAGNOSIS",
                        "candidate_source": candidate_source,
                        "candidate_step": candidate_step,
                        "residual_scale_multiplier": args.residual_scale_multiplier,
                        "selected_speeds": list(selected_speeds),
                        "completed_speeds": [float(value) for value in reports],
                        "long_steps": args.long_steps,
                        "speed_reports": reports,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if args.fail_fast and not all(long_checks.values()):
                break
            continue
        if args.disturbed_only:
            ppo_disturbed = _run_timed(
                speed,
                "ppo_disturbed",
                lambda: _evaluate_teacher(
                    jax,
                    disturbed_env,
                    teacher_policy,
                    args.seed + 500 + 1000 * index,
                    args.eval_envs,
                    args.steps,
                    **common_disturbed,
                ),
            )
            ik_disturbed = _run_timed(
                speed,
                "ik_disturbed",
                lambda: _evaluate_teacher(
                    jax,
                    disturbed_env,
                    zero_policy,
                    args.seed + 500 + 1000 * index,
                    args.eval_envs,
                    args.steps,
                    **common_disturbed,
                ),
            )
            disturbed_checks = _disturbed_gate_checks(
                speed, ppo_disturbed, ik_disturbed
            )
            _print_disturbed_gate(
                speed, disturbed_checks, ppo_disturbed, ik_disturbed
            )
            reports[f"{speed:.2f}"] = {
                "accepted": bool(all(disturbed_checks.values())),
                "command_vx": speed,
                "checks": disturbed_checks,
                "ppo_disturbed": ppo_disturbed,
                "ik_disturbed": ik_disturbed,
            }
            disturbed_diagnosis_path.write_text(
                json.dumps(
                    {
                        "stage": "T9_FORWARD_COMMAND_TEACHER_DISTURBED_DIAGNOSIS",
                        "candidate_source": candidate_source,
                        "candidate_step": candidate_step,
                        "selected_speeds": list(selected_speeds),
                        "completed_speeds": [float(value) for value in reports],
                        "steps": args.steps,
                        "speed_reports": reports,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                "stage=t9_teacher_disturbed_diagnosis_saved "
                f"completed={len(reports)}/{len(selected_speeds)} "
                f"report={disturbed_diagnosis_path}",
                flush=True,
            )
            if args.fail_fast and not all(disturbed_checks.values()):
                break
            continue
        ppo_nominal = _run_timed(
            speed,
            "ppo_nominal",
            lambda: _evaluate_teacher(
                jax,
                nominal_env,
                teacher_policy,
                args.seed + 1000 * index,
                args.eval_envs,
                args.steps,
                **common_nominal,
            ),
        )
        ik_nominal = _run_timed(
            speed,
            "ik_nominal",
            lambda: _evaluate_teacher(
                jax,
                nominal_env,
                zero_policy,
                args.seed + 1000 * index,
                args.eval_envs,
                args.steps,
                **common_nominal,
            ),
        )
        ppo_disturbed = _run_timed(
            speed,
            "ppo_disturbed",
            lambda: _evaluate_teacher(
                jax,
                disturbed_env,
                teacher_policy,
                args.seed + 500 + 1000 * index,
                args.eval_envs,
                args.steps,
                **common_disturbed,
            ),
        )
        ik_disturbed = _run_timed(
            speed,
            "ik_disturbed",
            lambda: _evaluate_teacher(
                jax,
                disturbed_env,
                zero_policy,
                args.seed + 500 + 1000 * index,
                args.eval_envs,
                args.steps,
                **common_disturbed,
            ),
        )
        ppo_trace = _run_timed(
            speed,
            "ppo_long",
            lambda: _rollout_policy(
                jax,
                jp,
                nominal_env,
                "teacher",
                teacher_policy,
                None,
                args.seed + 700 + 1000 * index,
                args.long_envs,
                args.long_steps,
                **common_nominal,
            ),
        )
        ik_trace = _run_timed(
            speed,
            "ik_long",
            lambda: _rollout_policy(
                jax,
                jp,
                nominal_env,
                "ik",
                teacher_policy,
                None,
                args.seed + 700 + 1000 * index,
                args.long_envs,
                args.long_steps,
                **common_nominal,
            ),
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
        disturbed_checks = gate["checks"]["disturbed"]
        _print_disturbed_gate(speed, disturbed_checks, ppo_disturbed, ik_disturbed)
        print(
            f"stage=t9_teacher_speed_gate vx={speed:.2f} accepted={gate['accepted']} "
            f"nominal={gate['nominal_preserved']} disturbed={gate['disturbed_preserved_or_improved']} "
            f"long={gate['long_horizon_safe']} lateral30={long_ppo['mean_absolute_lateral_displacement_m']:.4f}",
            flush=True,
        )

        partial_report = {
            "stage": "T9_FORWARD_COMMAND_TEACHER_GRID_EVALUATION_PARTIAL",
            "complete": False,
            "accepted": False,
            "candidate_source": candidate_source,
            "candidate_step": candidate_step,
            "candidate_params": str(candidate_params_path),
            "anchors": list(anchors),
            "selected_speeds": list(selected_speeds),
            "completed_speeds": [float(value) for value in reports],
            "steps": args.steps,
            "long_steps": args.long_steps,
            "speed_reports": reports,
        }
        partial_path.write_text(
            json.dumps(partial_report, indent=2), encoding="utf-8"
        )
        print(
            f"stage=t9_teacher_grid_partial_saved completed={len(reports)}/"
            f"{len(selected_speeds)} report={partial_path}",
            flush=True,
        )
        if args.fail_fast and not gate["accepted"]:
            print(
                f"stage=t9_teacher_grid_stopped reason=failed_speed vx={speed:.2f}",
                flush=True,
            )
            break

    if args.long_only:
        failed = any(not report["accepted"] for report in reports.values())
        print(
            "stage=t9_teacher_long_diagnosis_complete "
            f"failed={failed} report={long_diagnosis_path}",
            flush=True,
        )
        if args.strict and failed:
            raise SystemExit(2)
        return

    if args.disturbed_only:
        failed = any(not report["accepted"] for report in reports.values())
        print(
            "stage=t9_teacher_disturbed_diagnosis_complete "
            f"failed={failed} report={disturbed_diagnosis_path}",
            flush=True,
        )
        if args.strict and failed:
            raise SystemExit(2)
        return

    full_grid_complete = len(reports) == len(anchors) and all(
        f"{speed:.2f}" in reports for speed in anchors
    )
    accepted = full_grid_complete and all(
        report["accepted"] for report in reports.values()
    )
    if not full_grid_complete:
        any_failed = any(not report["accepted"] for report in reports.values())
        print(
            "stage=t9_teacher_grid_diagnostic_complete "
            f"full_grid=False failed={any_failed} report={partial_path}",
            flush=True,
        )
        if args.strict and any_failed:
            raise SystemExit(2)
        return
    grid_report = {
        "stage": "T9_FORWARD_COMMAND_TEACHER_GRID_EVALUATION",
        "accepted": accepted,
        "candidate_source": candidate_source,
        "candidate_step": candidate_step,
        "candidate_params": str(candidate_params_path),
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
        params_path = teacher_run / "teacher" / "params"
        model_io.save_params(params_path, candidate_params)
        selection["selected_source"] = "ppo"
        selection["selected_step"] = candidate_step
        selection["selected_score"] = selection.get("ppo_score")
        selection["selected_by"] = "t9_speed_grid_gate"
        selection["selected_params"] = str(params_path)
        (teacher_run / "teacher" / "selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8"
        )
        evaluation["selected_source"] = "ppo"
        evaluation["selected_step"] = candidate_step
        evaluation["selected_by"] = "t9_speed_grid_gate"
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
