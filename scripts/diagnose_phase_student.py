from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from disk_robot.ik_reference import build_ik_reference
from disk_robot.student_policy import load_student_policy
from disk_robot_mjx.pipeline import configure_cloud_runtime, make_network_factory
from disk_robot_mjx.teacher_student_env import make_forward_teacher_student_env
from scripts.distill_forward_student import (
    _config_from_teacher_run,
    _load_accepted_teacher_run,
    _reference_spec_from_teacher_run,
    _resolve_xml_path,
)
from scripts.train_forward_teacher_student import (
    _evaluate_oracle_student,
    _evaluate_student,
    _evaluate_teacher,
    _print_evaluation_summary,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare frozen Teacher residual control, oracle direct actions, and T5 Student."
    )
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--student-run", type=Path, required=True)
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=120_000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mujoco-gl", default="egl")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    teacher_run, run_config, teacher_evaluation, _, params_path = (
        _load_accepted_teacher_run(args.teacher_run)
    )
    reference_spec = _reference_spec_from_teacher_run(run_config)
    base_config = _config_from_teacher_run(run_config)
    phase_config = replace(
        base_config,
        student_phase_conditioned=True,
        student_phase_frequency=reference_spec.frequency,
    )
    xml_path = _resolve_xml_path(run_config, args.xml_path)
    student_path = args.student_run.expanduser().resolve() / "student_policy_phase_bc.npz"
    if not student_path.exists():
        raise SystemExit(f"T5 Student policy is missing: {student_path}")

    configure_cloud_runtime(mujoco_gl=args.mujoco_gl, verbose=True)
    try:
        import jax
        import jax.numpy as jp
        from brax.io import model as model_io
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise SystemExit(f"Phase Student diagnosis requires the mjx312 stack: {exc}") from exc

    reference = build_ik_reference(xml_path, reference_spec)
    nominal_config = replace(phase_config, disturbance_enabled=False)
    teacher_env = make_forward_teacher_student_env(
        "teacher",
        config=nominal_config,
        reference=reference,
        xml_path=xml_path,
        seed=args.seed,
    )
    oracle_env = make_forward_teacher_student_env(
        "dagger",
        config=nominal_config,
        reference=reference,
        xml_path=xml_path,
        seed=args.seed,
    )
    student_env = make_forward_teacher_student_env(
        "student",
        config=nominal_config,
        xml_path=xml_path,
        seed=args.seed,
    )

    teacher_networks = make_network_factory(
        run_config.get("teacher_hidden", [256, 256, 128]), "elu"
    )(
        observation_size=teacher_env.observation_size,
        action_size=teacher_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    teacher_policy = ppo_networks.make_inference_fn(teacher_networks)(
        model_io.load_params(params_path), deterministic=True
    )
    artifact = load_student_policy(student_path)
    student_params = tuple(
        (jp.asarray(weight), jp.asarray(bias)) for weight, bias in artifact.params
    )
    obs_mean = jp.asarray(artifact.obs_mean)
    obs_std = jp.asarray(artifact.obs_std)

    teacher_report = _evaluate_teacher(
        jax, teacher_env, teacher_policy, args.seed, args.envs, args.steps
    )
    oracle_report = _evaluate_oracle_student(
        jax, oracle_env, teacher_policy, args.seed, args.envs, args.steps
    )
    student_report = _evaluate_student(
        jax,
        jp,
        student_env,
        student_params,
        obs_mean,
        obs_std,
        args.seed,
        args.envs,
        args.steps,
    )
    _print_evaluation_summary("phase_diagnosis", teacher_report, "teacher_residual")
    _print_evaluation_summary("phase_diagnosis", oracle_report, "oracle_direct")
    _print_evaluation_summary("phase_diagnosis", student_report, "learned_student")
    report = {
        "teacher_run": str(teacher_run),
        "student_policy": str(student_path),
        "teacher_reference": teacher_evaluation["nominal_evaluation"],
        "teacher_residual": teacher_report,
        "oracle_direct": oracle_report,
        "learned_student": student_report,
        "oracle_preserves_teacher_velocity": (
            oracle_report["mean_velocity_x"] >= teacher_report["mean_velocity_x"] - 0.01
        ),
    }
    output_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else args.student_run.expanduser().resolve() / "phase_diagnosis.json"
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "stage=phase_diagnosis "
        f"oracle_preserves_teacher_velocity={report['oracle_preserves_teacher_velocity']} "
        f"teacher_vx={teacher_report['mean_velocity_x']:.4f} "
        f"oracle_vx={oracle_report['mean_velocity_x']:.4f} "
        f"student_vx={student_report['mean_velocity_x']:.4f} "
        f"report={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
