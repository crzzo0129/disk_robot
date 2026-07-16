from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

from disk_robot.ik_reference import build_ik_reference
from disk_robot.student_policy import load_student_policy
from disk_robot.teacher_student_config import ForwardTeacherStudentConfig
from disk_robot_mjx.pipeline import configure_cloud_runtime, make_network_factory
from disk_robot_mjx.teacher_student_env import make_forward_teacher_student_env
from scripts.distill_forward_student import (
    _bc_acceptance,
    _config_from_teacher_run,
    _load_accepted_teacher_run,
    _print_retention,
    _reference_spec_from_teacher_run,
    _resolve_xml_path,
)
from scripts.train_forward_teacher_student import (
    _collect_dagger_dataset,
    _evaluate_student,
    _print_evaluation_summary,
    _save_student_policy,
    _train_student,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Use DAgger to correct closed-loop drift in a T3 BC Student or "
            "a T5 phase-conditioned BC Student."
        )
    )
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--bc-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-samples", type=int, default=65_536)
    parser.add_argument("--nominal-fraction", type=float, default=0.50)
    parser.add_argument("--rollout-envs", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=500)
    parser.add_argument("--dagger-updates", type=int, default=5_000)
    parser.add_argument("--student-batch-size", type=int, default=1024)
    parser.add_argument("--student-learning-rate", type=float, default=1e-5)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--teacher-rollout-blend-start", type=float, default=0.50)
    parser.add_argument("--teacher-rollout-blend-end", type=float, default=0.20)

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
    parser.add_argument(
        "--require-phase-conditioned",
        action="store_true",
        help="Reject phase-free T3 inputs; used by the dedicated T6 entrypoint.",
    )

    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--no-xla-triton", dest="xla_triton", action="store_false")
    parser.set_defaults(xla_triton=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _load_bc_run(bc_run: Path):
    bc_run = bc_run.expanduser().resolve()
    candidates = (
        ("student_policy_phase_bc.npz", "student_phase_bc_dataset.npz"),
        ("student_policy_bc.npz", "student_bc_dataset.npz"),
    )
    selected = next(
        (
            (bc_run / policy_name, bc_run / dataset_name)
            for policy_name, dataset_name in candidates
            if (bc_run / policy_name).exists()
        ),
        None,
    )
    if selected is None:
        expected = " or ".join(policy_name for policy_name, _ in candidates)
        raise SystemExit(f"BC policy is missing in {bc_run}; expected {expected}")
    policy_path, dataset_path = selected
    evaluation_path = bc_run / "evaluation.json"
    if not dataset_path.exists():
        raise SystemExit(
            f"DAgger requires the saved BC dataset: {dataset_path}. "
            "The BC run must use --save-dataset."
        )
    if not evaluation_path.exists():
        raise SystemExit(f"BC evaluation is missing: {evaluation_path}")
    artifact = load_student_policy(policy_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    with np.load(dataset_path) as archive:
        observations = archive["observations"].astype(np.float32, copy=True)
        actions = archive["actions"].astype(np.float32, copy=True)
    if len(observations) != len(actions) or len(observations) == 0:
        raise SystemExit("BC dataset observations/actions are empty or misaligned")
    return bc_run, policy_path, artifact, observations, actions, evaluation


def _config_from_student_artifact(artifact, teacher_config):
    stage = artifact.metadata.get("stage")
    if stage == "T3_BC":
        return teacher_config
    if stage != "T5_PHASE_BC":
        raise SystemExit(
            "DAgger requires a Student artifact with stage=T3_BC or stage=T5_PHASE_BC"
        )
    values = artifact.metadata.get("config")
    if not isinstance(values, dict):
        raise SystemExit("T5 Student metadata has no complete config contract")
    allowed = {field.name for field in fields(ForwardTeacherStudentConfig)}
    values = {key: value for key, value in values.items() if key in allowed}
    for key in ("student_action_scale", "residual_scale"):
        if key in values:
            values[key] = tuple(values[key])
    try:
        config = ForwardTeacherStudentConfig(**values)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"T5 Student config contract is invalid: {exc}") from exc
    if not config.student_phase_conditioned:
        raise SystemExit("T5 Student config must enable phase conditioning")
    return config


def _validate_bc_teacher_contract(
    artifact, teacher_run, teacher_evaluation, config, teacher_config
):
    metadata = artifact.metadata
    stage = metadata.get("stage")
    if stage not in {"T3_BC", "T5_PHASE_BC"}:
        raise SystemExit(
            "DAgger requires a Student artifact with stage=T3_BC or stage=T5_PHASE_BC"
        )
    if int(metadata.get("observation_size", -1)) != config.student_policy_observation_size:
        raise SystemExit("Student observation size does not match its config contract")
    if artifact.obs_mean.shape != (config.student_policy_observation_size,):
        raise SystemExit("Student normalization statistics do not match its observation size")
    if not artifact.params:
        raise SystemExit("Student network has no trainable layers")
    if artifact.params[0][0].shape[0] != config.student_policy_observation_size:
        raise SystemExit("Student network input does not match its observation size")
    if artifact.params[-1][1].shape[0] != config.action_size:
        raise SystemExit("Student network output does not match its action size")
    if int(metadata.get("action_size", -1)) != config.action_size:
        raise SystemExit("Student action size does not match the Teacher run")
    if int(metadata.get("teacher_selected_step", -1)) != int(
        teacher_evaluation["selected_step"]
    ):
        raise SystemExit("Student and frozen Teacher selected steps do not match")
    recorded_teacher = metadata.get("teacher_run")
    if recorded_teacher and Path(recorded_teacher).name != teacher_run.name:
        raise SystemExit("Student was distilled from a different Teacher run")
    for name in (
        "command_vx",
        "actuator_kp",
        "actuator_kd",
        "torque_limit",
        "startup_blend_steps",
        "disturbance_enabled",
    ):
        if getattr(config, name) != getattr(teacher_config, name):
            raise SystemExit(f"Student and Teacher configs disagree on {name}")
    if stage == "T5_PHASE_BC":
        oscillator = metadata.get("internal_oscillator", {})
        if not oscillator.get("enabled", False):
            raise SystemExit("T5 Student metadata must enable the internal oscillator")
        if config.student_internal_state_size != 3:
            raise SystemExit("T5 Student must expose three controller-owned phase values")


def _dagger_variant(artifact):
    if artifact.metadata.get("stage") == "T5_PHASE_BC":
        return {
            "stage_name": "T6_PHASE_DAGGER",
            "terminal_stage": "t6",
            "train_stage": "student_phase_dagger",
            "default_run": "student_t6_phase_dagger",
            "policy_name": "student_policy_phase_dagger.npz",
            "round_policy_prefix": "student_policy_phase_dagger_round_",
            "dataset_name": "student_phase_dagger_dataset.npz",
        }
    return {
        "stage_name": "T4_DAGGER",
        "terminal_stage": "t4",
        "train_stage": "student_dagger",
        "default_run": "student_t4_dagger",
        "policy_name": "student_policy_dagger.npz",
        "round_policy_prefix": "student_policy_dagger_round_",
        "dataset_name": "student_dagger_dataset.npz",
    }


def _student_score(nominal, disturbed):
    return (
        -40.0 * nominal.get("mean_velocity_error", 0.0)
        - 40.0 * disturbed.get("mean_velocity_error", 0.0)
        - 5.0 * nominal.get("failure_rate", 0.0)
        - 5.0 * disturbed.get("failure_rate", 0.0)
        - 0.5 * nominal.get("mean_roll_pitch_rate_rms", 0.0)
        - 0.5 * disturbed.get("mean_roll_pitch_rate_rms", 0.0)
        - disturbed.get("mean_post_push_velocity_error", 0.0)
        - 0.5 * disturbed.get("mean_recovery_time", 0.0)
        - disturbed.get("mean_disk_contacts", 0.0)
    )


def _teacher_rollout_blend(round_index, total_rounds, start, end):
    if total_rounds <= 1:
        return float(start)
    fraction = (round_index - 1) / (total_rounds - 1)
    return float(start + fraction * (end - start))


def _evaluate_round(
    jax,
    jp,
    nominal_env,
    disturbed_env,
    params,
    obs_mean,
    obs_std,
    seed,
    env_count,
    horizon,
    teacher_evaluation,
    args,
    round_index,
    terminal_stage,
    train_stage,
):
    nominal = _evaluate_student(
        jax,
        jp,
        nominal_env,
        params,
        obs_mean,
        obs_std,
        seed,
        env_count,
        horizon,
    )
    disturbed = _evaluate_student(
        jax,
        jp,
        disturbed_env,
        params,
        obs_mean,
        obs_std,
        seed + 1_000,
        env_count,
        horizon,
    )
    gate = _bc_acceptance(nominal, disturbed, teacher_evaluation, args)
    score = _student_score(nominal, disturbed)
    _print_evaluation_summary(
        f"{train_stage}_result round={round_index}", nominal, "nominal"
    )
    _print_evaluation_summary(
        f"{train_stage}_result round={round_index}", disturbed, "disturbed"
    )
    _print_retention(
        "nominal",
        nominal,
        teacher_evaluation["nominal_evaluation"],
        f"{train_stage}_retention",
    )
    _print_retention(
        "disturbed",
        disturbed,
        teacher_evaluation["disturbed_evaluation"],
        f"{train_stage}_retention",
    )
    print(
        f"stage={terminal_stage}_round round={round_index} score={score:.5f} "
        f"accepted={gate['accepted']} nominal_preserved={gate['nominal_preserved']} "
        f"disturbed_preserved={gate['disturbed_preserved']}",
        flush=True,
    )
    return {
        **gate,
        "round": int(round_index),
        "score": float(score),
        "nominal_evaluation": nominal,
        "disturbed_evaluation": disturbed,
    }


def _policy_metadata(
    config,
    reference_spec,
    teacher_run_config,
    teacher_run,
    teacher_evaluation,
    hidden_layers,
    round_report,
    rounds,
    stage_name,
    source_stage,
):
    return {
        "format": "disk_robot_student_mlp_v1",
        "stage": stage_name,
        "stand_source": "xml:keyframe:stand",
        "observation_size": config.student_policy_observation_size,
        "sensor_history_size": config.student_observation_size,
        "internal_state_size": config.student_internal_state_size,
        "observation_contract": (
            "192 sensor-history values followed by sin_phase, cos_phase, gait_blend"
            if config.student_phase_conditioned
            else "192 sensor-history values"
        ),
        "action_size": config.action_size,
        "hidden_layers": list(hidden_layers),
        "action_semantics": "q_target = q_stand + student_action_scale * tanh(policy)",
        "student_action_scale": list(config.student_action_scale),
        "command": [config.command_vx, 0.0, 0.0],
        "internal_oscillator": {
            "enabled": config.student_phase_conditioned,
            "frequency_hz": config.student_phase_frequency,
            "observation": ["sin_phase", "cos_phase", "gait_blend"]
            if config.student_phase_conditioned
            else [],
            "requires_foot_contact": False,
            "requires_ik_at_runtime": False,
        },
        "config": asdict(config),
        "ik_reference": asdict(reference_spec),
        "ik_reference_source": teacher_run_config.get("ik_reference_source", {}),
        "teacher_run": str(teacher_run),
        "teacher_source": "ppo",
        "teacher_selected_step": int(teacher_evaluation["selected_step"]),
        "source_student_stage": source_stage,
        "selected_dagger_round": int(round_report["round"]),
        "evaluation": round_report,
        "rounds": rounds,
    }


def main(argv=None):
    args = parse_args(argv)
    if args.dagger_rounds < 1:
        raise SystemExit("--dagger-rounds must be at least 1")
    if args.dagger_samples < 2:
        raise SystemExit("--dagger-samples must be at least 2")
    if not 0.0 < args.nominal_fraction < 1.0:
        raise SystemExit("--nominal-fraction must be in (0, 1)")
    if not 0.0 <= args.teacher_rollout_blend_end <= args.teacher_rollout_blend_start <= 1.0:
        raise SystemExit(
            "Teacher rollout blend must satisfy 0 <= end <= start <= 1"
        )
    if min(args.rollout_envs, args.rollout_horizon, args.dagger_updates, args.eval_envs) < 1:
        raise SystemExit("rollout, update, and evaluation counts must be positive")
    if args.anchor_weight < 0.0:
        raise SystemExit("--anchor-weight must be non-negative")

    teacher_run, teacher_run_config, teacher_evaluation, _, params_path = (
        _load_accepted_teacher_run(args.teacher_run)
    )
    teacher_config = _config_from_teacher_run(teacher_run_config)
    if not teacher_config.disturbance_enabled:
        raise SystemExit("DAgger requires a T2 Teacher trained with disturbances enabled")
    reference_spec = _reference_spec_from_teacher_run(teacher_run_config)
    xml_path = _resolve_xml_path(teacher_run_config, args.xml_path)
    (
        bc_run,
        bc_policy_path,
        bc_artifact,
        all_observations,
        all_labels,
        bc_evaluation,
    ) = _load_bc_run(args.bc_run)
    config = _config_from_student_artifact(bc_artifact, teacher_config)
    _validate_bc_teacher_contract(
        bc_artifact, teacher_run, teacher_evaluation, config, teacher_config
    )
    if all_observations.ndim != 2 or all_observations.shape[1] != (
        config.student_policy_observation_size
    ):
        raise SystemExit("Saved BC observations do not match the Student observation contract")
    if all_labels.ndim != 2 or all_labels.shape[1] != config.action_size:
        raise SystemExit("Saved BC labels do not match the Student action contract")
    variant = _dagger_variant(bc_artifact)
    if args.require_phase_conditioned and not config.student_phase_conditioned:
        raise SystemExit("T6 requires a phase-conditioned T5 Student run")
    if args.out is None:
        args.out = bc_run.parent / f"{variant['default_run']}_seed{args.seed}"
    args.out = args.out.expanduser().resolve()

    if args.smoke:
        args.dagger_rounds = 1
        args.dagger_samples = min(args.dagger_samples, 2_048)
        args.rollout_envs = min(args.rollout_envs, 16)
        args.rollout_horizon = min(args.rollout_horizon, 128)
        args.dagger_updates = min(args.dagger_updates, 20)
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
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise SystemExit(
            f"{variant['terminal_stage'].upper()} could not import the MJX stack: "
            f"{exc.name or exc}. Activate the offline mjx312 environment."
        ) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    reference = build_ik_reference(xml_path, reference_spec)
    nominal_config = replace(config, disturbance_enabled=False)
    nominal_dagger_env = make_forward_teacher_student_env(
        "dagger",
        config=nominal_config,
        reference=reference,
        xml_path=xml_path,
        seed=args.seed + 10_000,
    )
    disturbed_dagger_env = make_forward_teacher_student_env(
        "dagger",
        config=config,
        reference=reference,
        xml_path=xml_path,
        seed=args.seed + 20_000,
    )
    nominal_student_env = make_forward_teacher_student_env(
        "student", config=nominal_config, xml_path=xml_path, seed=args.seed + 30_000
    )
    disturbed_student_env = make_forward_teacher_student_env(
        "student", config=config, xml_path=xml_path, seed=args.seed + 40_000
    )

    teacher_networks = make_network_factory(
        teacher_run_config.get("teacher_hidden", [256, 256, 128]), "elu"
    )(
        observation_size=disturbed_dagger_env.config.teacher_observation_size,
        action_size=disturbed_dagger_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    teacher_policy = ppo_networks.make_inference_fn(teacher_networks)(
        model_io.load_params(params_path), deterministic=True
    )
    student_params = tuple(
        (jp.asarray(weight), jp.asarray(bias)) for weight, bias in bc_artifact.params
    )
    obs_mean = jp.asarray(bc_artifact.obs_mean)
    obs_std = jp.asarray(bc_artifact.obs_std)
    hidden_layers = bc_artifact.metadata.get("hidden_layers", [256, 128, 128])
    print(
        f"stage={variant['terminal_stage']}_source "
        f"teacher_step={int(teacher_evaluation['selected_step']):,} "
        f"bc_policy={bc_policy_path} bc_samples={len(all_observations):,} "
        f"rounds={args.dagger_rounds} obs={config.student_policy_observation_size} "
        f"phase_conditioned={config.student_phase_conditioned}",
        flush=True,
    )

    round_reports = []
    paired_evaluation_seed = args.seed + 50_000
    initial_report = _evaluate_round(
        jax,
        jp,
        nominal_student_env,
        disturbed_student_env,
        student_params,
        obs_mean,
        obs_std,
        paired_evaluation_seed,
        args.eval_envs,
        config.max_episode_steps,
        teacher_evaluation,
        args,
        0,
        variant["terminal_stage"],
        variant["train_stage"],
    )
    initial_report["dataset_samples"] = int(len(all_observations))
    initial_report["teacher_rollout_blend"] = 0.0
    round_reports.append(initial_report)
    best_report = initial_report
    best_params = student_params

    for round_index in range(1, args.dagger_rounds + 1):
        teacher_blend = _teacher_rollout_blend(
            round_index,
            args.dagger_rounds,
            args.teacher_rollout_blend_start,
            args.teacher_rollout_blend_end,
        )
        nominal_samples = int(round(args.dagger_samples * args.nominal_fraction))
        nominal_samples = min(max(nominal_samples, 1), args.dagger_samples - 1)
        disturbed_samples = args.dagger_samples - nominal_samples
        print(
            f"stage={variant['terminal_stage']}_dataset_plan round={round_index} "
            f"total={args.dagger_samples:,} "
            f"nominal={nominal_samples:,} disturbed={disturbed_samples:,} "
            f"teacher_blend={teacher_blend:.2f}",
            flush=True,
        )
        print(
            f"stage={variant['terminal_stage']}_dataset round={round_index} "
            "source=nominal status=collecting",
            flush=True,
        )
        nominal_obs, nominal_labels = _collect_dagger_dataset(
            jax,
            jp,
            nominal_dagger_env,
            teacher_policy,
            student_params,
            obs_mean,
            obs_std,
            args.seed + 60_000 + 10_000 * round_index,
            args.rollout_envs,
            args.rollout_horizon,
            nominal_samples,
            teacher_blend,
        )
        print(
            f"stage={variant['terminal_stage']}_dataset round={round_index} "
            "source=disturbed status=collecting",
            flush=True,
        )
        disturbed_obs, disturbed_labels = _collect_dagger_dataset(
            jax,
            jp,
            disturbed_dagger_env,
            teacher_policy,
            student_params,
            obs_mean,
            obs_std,
            args.seed + 70_000 + 10_000 * round_index,
            args.rollout_envs,
            args.rollout_horizon,
            disturbed_samples,
            teacher_blend,
        )
        all_observations = np.concatenate(
            (all_observations, nominal_obs, disturbed_obs)
        ).astype(np.float32, copy=False)
        all_labels = np.concatenate((all_labels, nominal_labels, disturbed_labels)).astype(
            np.float32, copy=False
        )
        round_anchor_params = student_params
        student_params = _train_student(
            jax,
            jp,
            optax,
            student_params,
            all_observations,
            all_labels,
            obs_mean,
            obs_std,
            args.dagger_updates,
            args.student_batch_size,
            args.student_learning_rate,
            args.seed + 80_000 + round_index,
            f"{variant['train_stage']}_{round_index}",
            round_anchor_params,
            args.anchor_weight,
        )
        report = _evaluate_round(
            jax,
            jp,
            nominal_student_env,
            disturbed_student_env,
            student_params,
            obs_mean,
            obs_std,
            paired_evaluation_seed,
            args.eval_envs,
            config.max_episode_steps,
            teacher_evaluation,
            args,
            round_index,
            variant["terminal_stage"],
            variant["train_stage"],
        )
        report["dataset_samples"] = int(len(all_observations))
        report["teacher_rollout_blend"] = float(teacher_blend)
        round_reports.append(report)
        round_path = (
            args.out / f"{variant['round_policy_prefix']}{round_index}.npz"
        )
        _save_student_policy(
            round_path,
            student_params,
            obs_mean,
            obs_std,
            _policy_metadata(
                config,
                reference_spec,
                teacher_run_config,
                teacher_run,
                teacher_evaluation,
                hidden_layers,
                report,
                round_reports,
                variant["stage_name"],
                bc_artifact.metadata.get("stage"),
            ),
        )
        if (report["accepted"] and not best_report["accepted"]) or (
            report["accepted"] == best_report["accepted"]
            and report["score"] > best_report["score"]
        ):
            best_report = report
            best_params = student_params

    if args.save_dataset:
        np.savez_compressed(
            args.out / variant["dataset_name"],
            observations=all_observations,
            actions=all_labels,
        )
    final_report = {
        "stage": variant["stage_name"],
        "phase_conditioned": bool(config.student_phase_conditioned),
        "observation_size": int(config.student_policy_observation_size),
        "accepted": bool(best_report["accepted"]),
        "selected_round": int(best_report["round"]),
        "selected_score": float(best_report["score"]),
        "teacher_run": str(teacher_run),
        "bc_run": str(bc_run),
        "initial_bc_evaluation": bc_evaluation,
        "selected_evaluation": best_report,
        "rounds": round_reports,
    }
    policy_path = args.out / variant["policy_name"]
    _save_student_policy(
        policy_path,
        best_params,
        obs_mean,
        obs_std,
        _policy_metadata(
            config,
            reference_spec,
            teacher_run_config,
            teacher_run,
            teacher_evaluation,
            hidden_layers,
            best_report,
            round_reports,
            variant["stage_name"],
            bc_artifact.metadata.get("stage"),
        ),
    )
    (args.out / "evaluation.json").write_text(
        json.dumps(final_report, indent=2), encoding="utf-8"
    )
    run_record = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    run_record.update(
        {
            "stage": variant["stage_name"],
            "resolved_xml_path": str(xml_path),
            "teacher_params": str(params_path),
            "bc_policy": str(bc_policy_path),
            "source_student_stage": bc_artifact.metadata.get("stage"),
            "config": asdict(config),
            "initial_dataset_samples": int(len(all_observations) - args.dagger_rounds * args.dagger_samples),
            "final_dataset_samples": int(len(all_observations)),
        }
    )
    (args.out / "run_config.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    print(
        f"stage={variant['terminal_stage']}_acceptance "
        f"accepted={best_report['accepted']} "
        f"selected_round={best_report['round']} score={best_report['score']:.5f} "
        f"policy={policy_path} report={args.out / 'evaluation.json'}",
        flush=True,
    )
    if args.strict_acceptance and not best_report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
