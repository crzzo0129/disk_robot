from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from disk_robot.ik_reference import IKReferenceSpec, build_ik_reference
from disk_robot.teacher_student_config import ForwardTeacherStudentConfig
from disk_robot_mjx.pipeline import configure_cloud_runtime, make_network_factory
from disk_robot_mjx.teacher_student_env import DEFAULT_XML, make_forward_teacher_student_env
from scripts.train_forward_teacher_student import (
    _collect_teacher_dataset,
    _evaluate_student,
    _print_evaluation_summary,
    _save_student_policy,
    _student_init,
    _train_student,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="T3: distill one accepted, frozen privileged Teacher into a gait-free BC Student."
    )
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--xml-path",
        type=Path,
        default=None,
        help="Override the Teacher run XML. By default, reuse its XML or the repository candidate XML.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-samples", type=int, default=131_072)
    parser.add_argument("--nominal-fraction", type=float, default=0.50)
    parser.add_argument("--rollout-envs", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=500)
    parser.add_argument("--student-hidden", type=int, nargs="+", default=[256, 128, 128])
    parser.add_argument("--student-updates", type=int, default=20_000)
    parser.add_argument("--student-batch-size", type=int, default=1024)
    parser.add_argument("--student-learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--save-dataset", action="store_true")

    parser.add_argument("--nominal-vx-tolerance", type=float, default=0.015)
    parser.add_argument("--nominal-failure-tolerance", type=float, default=0.05)
    parser.add_argument("--nominal-roll-pitch-tolerance", type=float, default=0.10)
    parser.add_argument("--nominal-lateral-tolerance", type=float, default=0.015)
    parser.add_argument("--nominal-yaw-tolerance", type=float, default=0.05)
    parser.add_argument("--disturbed-failure-tolerance", type=float, default=0.05)
    parser.add_argument("--disturbed-post-error-tolerance", type=float, default=0.03)
    parser.add_argument("--disturbed-recovery-tolerance", type=float, default=0.50)
    parser.add_argument("--disturbed-distance-tolerance", type=float, default=0.10)
    parser.add_argument("--disturbed-disk-tolerance", type=float, default=0.02)
    parser.add_argument("--strict-acceptance", action="store_true")

    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--no-xla-triton", dest="xla_triton", action="store_false")
    parser.set_defaults(xla_triton=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"required Teacher artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON Teacher artifact: {path}: {exc}") from exc


def _load_accepted_teacher_run(teacher_run: Path):
    teacher_run = teacher_run.expanduser().resolve()
    run_config = _read_json(teacher_run / "run_config.json")
    evaluation = _read_json(teacher_run / "teacher" / "evaluation.json")
    selection = _read_json(teacher_run / "teacher" / "selection.json")
    params_path = teacher_run / "teacher" / "params"
    if not evaluation.get("accepted", False):
        raise SystemExit("T3 requires a Teacher run with accepted=true")
    if evaluation.get("selected_source") != "ppo":
        raise SystemExit("T3 requires the accepted PPO Teacher, not the zero-residual IK baseline")
    if not params_path.exists():
        raise SystemExit(f"accepted Teacher parameters are missing: {params_path}")
    return teacher_run, run_config, evaluation, selection, params_path


def _config_from_teacher_run(run_config):
    defaults = ForwardTeacherStudentConfig()
    residual_multiplier = float(run_config.get("residual_scale_multiplier", 1.0))
    return ForwardTeacherStudentConfig(
        max_episode_steps=int(run_config.get("episode_length", defaults.max_episode_steps)),
        command_vx=float(run_config.get("command_vx", defaults.command_vx)),
        actuator_kp=float(run_config.get("kp", defaults.actuator_kp)),
        actuator_kd=float(run_config.get("kd", defaults.actuator_kd)),
        torque_limit=float(run_config.get("torque_limit", defaults.torque_limit)),
        startup_blend_steps=int(run_config.get("startup_steps", defaults.startup_blend_steps)),
        residual_filter_alpha=float(
            run_config.get("residual_filter_alpha", defaults.residual_filter_alpha)
        ),
        penalty_residual=float(run_config.get("penalty_residual", defaults.penalty_residual)),
        penalty_residual_rate=float(
            run_config.get("penalty_residual_rate", defaults.penalty_residual_rate)
        ),
        disturbance_enabled=bool(run_config.get("teacher_disturbances", False)),
        push_step_min=int(run_config.get("push_step_min", defaults.push_step_min)),
        push_step_max=int(run_config.get("push_step_max", defaults.push_step_max)),
        push_velocity_x=float(run_config.get("push_velocity_x", defaults.push_velocity_x)),
        push_velocity_y=float(run_config.get("push_velocity_y", defaults.push_velocity_y)),
        motor_strength_min=float(
            run_config.get("motor_strength_min", defaults.motor_strength_min)
        ),
        motor_strength_max=float(
            run_config.get("motor_strength_max", defaults.motor_strength_max)
        ),
        control_delay_probability=float(
            run_config.get("control_delay_probability", defaults.control_delay_probability)
        ),
        disturbance_reset_joint_noise=float(
            run_config.get(
                "disturbance_reset_joint_noise", defaults.disturbance_reset_joint_noise
            )
        ),
        disturbance_reset_height_noise=float(
            run_config.get(
                "disturbance_reset_height_noise", defaults.disturbance_reset_height_noise
            )
        ),
        recovery_window_steps=int(
            run_config.get("recovery_window_steps", defaults.recovery_window_steps)
        ),
        recovery_velocity_ema_alpha=float(
            run_config.get(
                "recovery_velocity_ema_alpha", defaults.recovery_velocity_ema_alpha
            )
        ),
        recovery_forward_tolerance=float(
            run_config.get(
                "recovery_forward_tolerance", defaults.recovery_forward_tolerance
            )
        ),
        recovery_lateral_tolerance=float(
            run_config.get(
                "recovery_lateral_tolerance", defaults.recovery_lateral_tolerance
            )
        ),
        recovery_required_steps=int(
            run_config.get("recovery_required_steps", defaults.recovery_required_steps)
        ),
        residual_scale=tuple(value * residual_multiplier for value in defaults.residual_scale),
    )


def _reference_spec_from_teacher_run(run_config):
    values = run_config.get("resolved_ik_reference")
    if not isinstance(values, dict):
        raise SystemExit("Teacher run_config.json has no resolved_ik_reference")
    return IKReferenceSpec(**values)


def _resolve_xml_path(run_config, override: Path | None):
    if override is not None:
        return override.expanduser().resolve()
    stored = run_config.get("xml_path")
    if stored:
        stored_path = Path(stored).expanduser()
        if stored_path.exists():
            return stored_path.resolve()
    return DEFAULT_XML.resolve()


def _bc_acceptance(nominal, disturbed, teacher_evaluation, args):
    teacher_nominal = teacher_evaluation["nominal_evaluation"]
    teacher_disturbed = teacher_evaluation["disturbed_evaluation"]
    nominal_checks = {
        "velocity_x": nominal["mean_velocity_x"]
        >= teacher_nominal["mean_velocity_x"] - args.nominal_vx_tolerance,
        "failure_rate": nominal["failure_rate"]
        <= teacher_nominal["failure_rate"] + args.nominal_failure_tolerance,
        "roll_pitch_rate": nominal["mean_roll_pitch_rate_rms"]
        <= teacher_nominal["mean_roll_pitch_rate_rms"] + args.nominal_roll_pitch_tolerance,
        "lateral_speed": nominal["mean_abs_velocity_y"]
        <= teacher_nominal["mean_abs_velocity_y"] + args.nominal_lateral_tolerance,
        "yaw_rate": nominal["mean_abs_yaw_rate"]
        <= teacher_nominal["mean_abs_yaw_rate"] + args.nominal_yaw_tolerance,
    }
    disturbed_checks = {
        "failure_rate": disturbed["failure_rate"]
        <= teacher_disturbed["failure_rate"] + args.disturbed_failure_tolerance,
        "post_push_error": disturbed["mean_post_push_velocity_error"]
        <= teacher_disturbed["mean_post_push_velocity_error"]
        + args.disturbed_post_error_tolerance,
        "recovery_time": disturbed["mean_recovery_time"]
        <= teacher_disturbed["mean_recovery_time"] + args.disturbed_recovery_tolerance,
        "forward_distance": disturbed["mean_forward_distance"]
        >= teacher_disturbed["mean_forward_distance"] - args.disturbed_distance_tolerance,
        "disk_contacts": disturbed["mean_disk_contacts"]
        <= teacher_disturbed["mean_disk_contacts"] + args.disturbed_disk_tolerance,
    }
    nominal_preserved = all(nominal_checks.values())
    disturbed_preserved = all(disturbed_checks.values())
    return {
        "accepted": nominal_preserved and disturbed_preserved,
        "nominal_preserved": nominal_preserved,
        "disturbed_preserved": disturbed_preserved,
        "nominal_checks": nominal_checks,
        "disturbed_checks": disturbed_checks,
    }


def _print_retention(mode, student, teacher):
    values = [
        f"delta_vx={student['mean_velocity_x'] - teacher['mean_velocity_x']:+.4f}",
        f"delta_failure={student['failure_rate'] - teacher['failure_rate']:+.3f}",
        f"delta_roll_pitch={student['mean_roll_pitch_rate_rms'] - teacher['mean_roll_pitch_rate_rms']:+.4f}",
    ]
    if mode == "disturbed":
        values.extend(
            (
                f"delta_post_error={student['mean_post_push_velocity_error'] - teacher['mean_post_push_velocity_error']:+.4f}",
                f"delta_recovery_s={student['mean_recovery_time'] - teacher['mean_recovery_time']:+.3f}",
                f"delta_distance={student['mean_forward_distance'] - teacher['mean_forward_distance']:+.4f}",
            )
        )
    print(f"stage=student_bc_retention mode={mode} {' '.join(values)}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.dataset_samples < 2:
        raise SystemExit("--dataset-samples must be at least 2")
    if not 0.0 < args.nominal_fraction < 1.0:
        raise SystemExit("--nominal-fraction must be in (0, 1)")
    if min(args.rollout_envs, args.rollout_horizon, args.student_updates, args.eval_envs) < 1:
        raise SystemExit("rollout, update, and evaluation counts must be positive")

    teacher_run, teacher_run_config, teacher_evaluation, selection, params_path = (
        _load_accepted_teacher_run(args.teacher_run)
    )
    config = _config_from_teacher_run(teacher_run_config)
    if not config.disturbance_enabled:
        raise SystemExit("T3 requires a T2 Teacher trained with disturbances enabled")
    reference_spec = _reference_spec_from_teacher_run(teacher_run_config)
    xml_path = _resolve_xml_path(teacher_run_config, args.xml_path)
    if not xml_path.exists():
        raise SystemExit(f"robot XML does not exist: {xml_path}")
    if args.out is None:
        args.out = teacher_run.parent / f"student_t3_bc_seed{args.seed}"
    args.out = args.out.expanduser().resolve()

    if args.smoke:
        args.dataset_samples = min(args.dataset_samples, 2_048)
        args.rollout_envs = min(args.rollout_envs, 16)
        args.rollout_horizon = min(args.rollout_horizon, 128)
        args.student_updates = min(args.student_updates, 20)
        args.student_batch_size = min(args.student_batch_size, 256)
        args.eval_envs = min(args.eval_envs, 16)

    configure_cloud_runtime(
        xla_triton=args.xla_triton,
        mujoco_gl=args.mujoco_gl,
        matmul_precision="high",
        verbose=True,
    )
    try:
        import jax
        import jax.numpy as jp
        import optax
        from brax.io import model as model_io
        from brax.training import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise SystemExit(
            "Activate the offline mjx312 environment with jax, mujoco, brax, and optax installed."
        ) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    reference = build_ik_reference(xml_path, reference_spec)
    nominal_config = replace(config, disturbance_enabled=False)
    nominal_teacher_env = make_forward_teacher_student_env(
        "teacher",
        config=nominal_config,
        reference=reference,
        xml_path=xml_path,
        seed=args.seed + 10_000,
    )
    disturbed_teacher_env = make_forward_teacher_student_env(
        "teacher",
        config=config,
        reference=reference,
        xml_path=xml_path,
        seed=args.seed + 20_000,
    )

    teacher_network_factory = make_network_factory(
        teacher_run_config.get("teacher_hidden", [256, 256, 128]),
        "elu",
    )
    teacher_networks = teacher_network_factory(
        observation_size=disturbed_teacher_env.observation_size,
        action_size=disturbed_teacher_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_teacher_policy = ppo_networks.make_inference_fn(teacher_networks)
    teacher_params = model_io.load_params(params_path)
    teacher_policy = make_teacher_policy(teacher_params, deterministic=True)
    print(
        f"stage=t3_teacher status=loaded source=ppo step={int(teacher_evaluation['selected_step']):,} "
        f"obs={disturbed_teacher_env.observation_size} action={disturbed_teacher_env.action_size} "
        f"params={params_path}",
        flush=True,
    )

    nominal_samples = int(round(args.dataset_samples * args.nominal_fraction))
    nominal_samples = min(max(nominal_samples, 1), args.dataset_samples - 1)
    disturbed_samples = args.dataset_samples - nominal_samples
    print(
        f"stage=t3_dataset_plan total={args.dataset_samples:,} nominal={nominal_samples:,} "
        f"disturbed={disturbed_samples:,} envs={args.rollout_envs} horizon={args.rollout_horizon}",
        flush=True,
    )
    print("stage=t3_dataset source=nominal status=collecting", flush=True)
    nominal_obs, nominal_labels = _collect_teacher_dataset(
        jax,
        jp,
        nominal_teacher_env,
        teacher_policy,
        args.seed + 30_000,
        args.rollout_envs,
        args.rollout_horizon,
        nominal_samples,
    )
    print("stage=t3_dataset source=disturbed status=collecting", flush=True)
    disturbed_obs, disturbed_labels = _collect_teacher_dataset(
        jax,
        jp,
        disturbed_teacher_env,
        teacher_policy,
        args.seed + 40_000,
        args.rollout_envs,
        args.rollout_horizon,
        disturbed_samples,
    )
    observations = np.concatenate((nominal_obs, disturbed_obs)).astype(np.float32)
    labels = np.concatenate((nominal_labels, disturbed_labels)).astype(np.float32)
    shuffle = np.random.default_rng(args.seed + 50_000).permutation(len(observations))
    observations = observations[shuffle]
    labels = labels[shuffle]
    obs_mean = jp.asarray(np.mean(observations, axis=0).astype(np.float32))
    obs_std = jp.asarray(np.maximum(np.std(observations, axis=0), 1e-3).astype(np.float32))

    layer_sizes = [config.student_observation_size, *args.student_hidden, config.action_size]
    student_params = _student_init(
        jax, jp, jax.random.PRNGKey(args.seed + 60_000), layer_sizes
    )
    student_params = _train_student(
        jax,
        jp,
        optax,
        student_params,
        observations,
        labels,
        obs_mean,
        obs_std,
        args.student_updates,
        args.student_batch_size,
        args.student_learning_rate,
        args.seed + 70_000,
        "student_bc",
    )
    if args.save_dataset:
        np.savez_compressed(
            args.out / "student_bc_dataset.npz",
            observations=observations,
            actions=labels,
            nominal_samples=np.asarray(nominal_samples),
            disturbed_samples=np.asarray(disturbed_samples),
        )

    nominal_student_env = make_forward_teacher_student_env(
        "student", config=nominal_config, xml_path=xml_path, seed=args.seed + 80_000
    )
    disturbed_student_env = make_forward_teacher_student_env(
        "student", config=config, xml_path=xml_path, seed=args.seed + 90_000
    )
    nominal_report = _evaluate_student(
        jax,
        jp,
        nominal_student_env,
        student_params,
        obs_mean,
        obs_std,
        args.seed + 100_000,
        args.eval_envs,
        config.max_episode_steps,
    )
    disturbed_report = _evaluate_student(
        jax,
        jp,
        disturbed_student_env,
        student_params,
        obs_mean,
        obs_std,
        args.seed + 110_000,
        args.eval_envs,
        config.max_episode_steps,
    )
    gate = _bc_acceptance(nominal_report, disturbed_report, teacher_evaluation, args)
    report = {
        **gate,
        "stage": "T3_BC",
        "teacher_run": str(teacher_run),
        "teacher_selected_step": int(teacher_evaluation["selected_step"]),
        "dataset_samples": int(args.dataset_samples),
        "nominal_samples": int(nominal_samples),
        "disturbed_samples": int(disturbed_samples),
        "nominal_evaluation": nominal_report,
        "disturbed_evaluation": disturbed_report,
        "teacher_nominal_evaluation": teacher_evaluation["nominal_evaluation"],
        "teacher_disturbed_evaluation": teacher_evaluation["disturbed_evaluation"],
    }
    metadata = {
        "format": "disk_robot_student_mlp_v1",
        "stage": "T3_BC",
        "xml_path": str(xml_path),
        "stand_source": "xml:keyframe:stand",
        "observation_size": config.student_observation_size,
        "action_size": config.action_size,
        "hidden_layers": args.student_hidden,
        "action_semantics": "q_target = q_stand + student_action_scale * tanh(policy)",
        "student_action_scale": list(config.student_action_scale),
        "command": [config.command_vx, 0.0, 0.0],
        "config": asdict(config),
        "ik_reference": asdict(reference_spec),
        "ik_reference_source": teacher_run_config.get("ik_reference_source", {}),
        "teacher_run": str(teacher_run),
        "teacher_source": "ppo",
        "teacher_selected_step": int(teacher_evaluation["selected_step"]),
        "teacher_selection": selection,
        "evaluation": report,
    }
    student_path = args.out / "student_policy_bc.npz"
    _save_student_policy(student_path, student_params, obs_mean, obs_std, metadata)
    (args.out / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    run_record = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    run_record.update(
        {
            "resolved_xml_path": str(xml_path),
            "teacher_selected_step": int(teacher_evaluation["selected_step"]),
            "teacher_params": str(params_path),
            "config": asdict(config),
            "resolved_ik_reference": asdict(reference_spec),
        }
    )
    (args.out / "run_config.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )

    _print_evaluation_summary("student_bc_result", nominal_report, "nominal")
    _print_evaluation_summary("student_bc_result", disturbed_report, "disturbed")
    _print_retention(
        "nominal", nominal_report, teacher_evaluation["nominal_evaluation"]
    )
    _print_retention(
        "disturbed", disturbed_report, teacher_evaluation["disturbed_evaluation"]
    )
    print(
        f"stage=t3_acceptance accepted={gate['accepted']} "
        f"nominal_preserved={gate['nominal_preserved']} "
        f"disturbed_preserved={gate['disturbed_preserved']} "
        f"policy={student_path} report={args.out / 'evaluation.json'}",
        flush=True,
    )
    if args.strict_acceptance and not gate["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
