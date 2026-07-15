from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from disk_robot.ik_reference import build_ik_reference
from disk_robot.student_policy import load_student_policy
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
        description="T4: use DAgger to correct closed-loop drift in a T3 BC Student."
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
    parser.add_argument("--student-learning-rate", type=float, default=1e-4)
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


def _load_bc_run(bc_run: Path):
    bc_run = bc_run.expanduser().resolve()
    policy_path = bc_run / "student_policy_bc.npz"
    dataset_path = bc_run / "student_bc_dataset.npz"
    evaluation_path = bc_run / "evaluation.json"
    if not policy_path.exists():
        raise SystemExit(f"T3 BC policy is missing: {policy_path}")
    if not dataset_path.exists():
        raise SystemExit(
            f"T4 requires the saved T3 dataset: {dataset_path}. T3 must use --save-dataset."
        )
    if not evaluation_path.exists():
        raise SystemExit(f"T3 evaluation is missing: {evaluation_path}")
    artifact = load_student_policy(policy_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    with np.load(dataset_path) as archive:
        observations = archive["observations"].astype(np.float32, copy=True)
        actions = archive["actions"].astype(np.float32, copy=True)
    if len(observations) != len(actions) or len(observations) == 0:
        raise SystemExit("T3 dataset observations/actions are empty or misaligned")
    return bc_run, policy_path, artifact, observations, actions, evaluation


def _validate_bc_teacher_contract(artifact, teacher_run, teacher_evaluation, config):
    metadata = artifact.metadata
    if metadata.get("stage") != "T3_BC":
        raise SystemExit("T4 requires a student_policy_bc artifact with stage=T3_BC")
    if int(metadata.get("observation_size", -1)) != config.student_observation_size:
        raise SystemExit("T3 Student observation size does not match the Teacher run")
    if int(metadata.get("action_size", -1)) != config.action_size:
        raise SystemExit("T3 Student action size does not match the Teacher run")
    if int(metadata.get("teacher_selected_step", -1)) != int(
        teacher_evaluation["selected_step"]
    ):
        raise SystemExit("T3 Student and frozen Teacher selected steps do not match")
    recorded_teacher = metadata.get("teacher_run")
    if recorded_teacher and Path(recorded_teacher).name != teacher_run.name:
        raise SystemExit("T3 Student was distilled from a different Teacher run")


def _student_score(nominal, disturbed):
    return (
        nominal.get("reward_per_step", 0.0)
        + disturbed.get("reward_per_step", 0.0)
        - 10.0 * nominal.get("mean_velocity_error", 0.0)
        - 10.0 * disturbed.get("mean_velocity_error", 0.0)
        - 3.0 * nominal.get("failure_rate", 0.0)
        - 3.0 * disturbed.get("failure_rate", 0.0)
        - 0.5 * nominal.get("mean_roll_pitch_rate_rms", 0.0)
        - 0.5 * disturbed.get("mean_roll_pitch_rate_rms", 0.0)
        - disturbed.get("mean_post_push_velocity_error", 0.0)
        - 0.5 * disturbed.get("mean_recovery_time", 0.0)
        - disturbed.get("mean_disk_contacts", 0.0)
    )


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
        f"student_dagger_result round={round_index}", nominal, "nominal"
    )
    _print_evaluation_summary(
        f"student_dagger_result round={round_index}", disturbed, "disturbed"
    )
    _print_retention(
        "nominal",
        nominal,
        teacher_evaluation["nominal_evaluation"],
        "student_dagger_retention",
    )
    _print_retention(
        "disturbed",
        disturbed,
        teacher_evaluation["disturbed_evaluation"],
        "student_dagger_retention",
    )
    print(
        f"stage=t4_round round={round_index} score={score:.5f} "
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
):
    return {
        "format": "disk_robot_student_mlp_v1",
        "stage": "T4_DAGGER",
        "stand_source": "xml:keyframe:stand",
        "observation_size": config.student_observation_size,
        "action_size": config.action_size,
        "hidden_layers": list(hidden_layers),
        "action_semantics": "q_target = q_stand + student_action_scale * tanh(policy)",
        "student_action_scale": list(config.student_action_scale),
        "command": [config.command_vx, 0.0, 0.0],
        "config": asdict(config),
        "ik_reference": asdict(reference_spec),
        "ik_reference_source": teacher_run_config.get("ik_reference_source", {}),
        "teacher_run": str(teacher_run),
        "teacher_source": "ppo",
        "teacher_selected_step": int(teacher_evaluation["selected_step"]),
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
    if min(args.rollout_envs, args.rollout_horizon, args.dagger_updates, args.eval_envs) < 1:
        raise SystemExit("rollout, update, and evaluation counts must be positive")

    teacher_run, teacher_run_config, teacher_evaluation, _, params_path = (
        _load_accepted_teacher_run(args.teacher_run)
    )
    config = _config_from_teacher_run(teacher_run_config)
    if not config.disturbance_enabled:
        raise SystemExit("T4 requires a T2 Teacher trained with disturbances enabled")
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
    _validate_bc_teacher_contract(bc_artifact, teacher_run, teacher_evaluation, config)
    if args.out is None:
        args.out = bc_run.parent / f"student_t4_dagger_seed{args.seed}"
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
            "T4 could not import the MJX stack: "
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
        f"stage=t4_source teacher_step={int(teacher_evaluation['selected_step']):,} "
        f"bc_policy={bc_policy_path} bc_samples={len(all_observations):,} "
        f"rounds={args.dagger_rounds}",
        flush=True,
    )

    round_reports = []
    initial_report = _evaluate_round(
        jax,
        jp,
        nominal_student_env,
        disturbed_student_env,
        student_params,
        obs_mean,
        obs_std,
        args.seed + 50_000,
        args.eval_envs,
        config.max_episode_steps,
        teacher_evaluation,
        args,
        0,
    )
    initial_report["dataset_samples"] = int(len(all_observations))
    round_reports.append(initial_report)
    best_report = initial_report
    best_params = student_params

    for round_index in range(1, args.dagger_rounds + 1):
        nominal_samples = int(round(args.dagger_samples * args.nominal_fraction))
        nominal_samples = min(max(nominal_samples, 1), args.dagger_samples - 1)
        disturbed_samples = args.dagger_samples - nominal_samples
        print(
            f"stage=t4_dataset_plan round={round_index} total={args.dagger_samples:,} "
            f"nominal={nominal_samples:,} disturbed={disturbed_samples:,}",
            flush=True,
        )
        print(
            f"stage=t4_dataset round={round_index} source=nominal status=collecting",
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
        )
        print(
            f"stage=t4_dataset round={round_index} source=disturbed status=collecting",
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
        )
        all_observations = np.concatenate(
            (all_observations, nominal_obs, disturbed_obs)
        ).astype(np.float32, copy=False)
        all_labels = np.concatenate((all_labels, nominal_labels, disturbed_labels)).astype(
            np.float32, copy=False
        )
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
            f"student_dagger_{round_index}",
        )
        report = _evaluate_round(
            jax,
            jp,
            nominal_student_env,
            disturbed_student_env,
            student_params,
            obs_mean,
            obs_std,
            args.seed + 90_000 + 2_000 * round_index,
            args.eval_envs,
            config.max_episode_steps,
            teacher_evaluation,
            args,
            round_index,
        )
        report["dataset_samples"] = int(len(all_observations))
        round_reports.append(report)
        round_path = args.out / f"student_policy_dagger_round_{round_index}.npz"
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
            args.out / "student_dagger_dataset.npz",
            observations=all_observations,
            actions=all_labels,
        )
    final_report = {
        "stage": "T4_DAGGER",
        "accepted": bool(best_report["accepted"]),
        "selected_round": int(best_report["round"]),
        "selected_score": float(best_report["score"]),
        "teacher_run": str(teacher_run),
        "bc_run": str(bc_run),
        "initial_bc_evaluation": bc_evaluation,
        "selected_evaluation": best_report,
        "rounds": round_reports,
    }
    policy_path = args.out / "student_policy_dagger.npz"
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
            "resolved_xml_path": str(xml_path),
            "teacher_params": str(params_path),
            "bc_policy": str(bc_policy_path),
            "initial_dataset_samples": int(len(all_observations) - args.dagger_rounds * args.dagger_samples),
            "final_dataset_samples": int(len(all_observations)),
        }
    )
    (args.out / "run_config.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    print(
        f"stage=t4_acceptance accepted={best_report['accepted']} "
        f"selected_round={best_report['round']} score={best_report['score']:.5f} "
        f"policy={policy_path} report={args.out / 'evaluation.json'}",
        flush=True,
    )
    if args.strict_acceptance and not best_report["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
