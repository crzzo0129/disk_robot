from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

from disk_robot.ik_reference import IKReferenceSpec, build_ik_reference_bank
from disk_robot.student_policy import load_student_policy, make_student_policy_jax
from disk_robot.t9_command import make_t9_config, validate_forward_speed_anchors
from disk_robot.teacher_student_config import ForwardTeacherStudentConfig
from disk_robot_mjx.pipeline import configure_cloud_runtime, make_network_factory
from disk_robot_mjx.teacher_student_env import make_forward_teacher_student_env
from scripts.characterize_t8_trajectories import _rollout_policy, summarize_trajectory
from scripts.distill_forward_student import _config_from_teacher_run, _resolve_xml_path
from scripts.train_forward_teacher_student import (
    _collect_teacher_dataset,
    _evaluate_teacher,
    _evaluate_student,
    _normalized_student_apply,
    _save_student_policy,
    _student_init,
    _train_student,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="T9 stage 2: BC-distill an accepted vx-grid Teacher into a 138D Student."
    )
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--t8-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-samples", type=int, default=196_608)
    parser.add_argument("--nominal-fraction", type=float, default=0.5)
    parser.add_argument("--rollout-envs", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=500)
    parser.add_argument("--student-hidden", type=int, nargs="+", default=[256, 128, 128])
    parser.add_argument("--student-updates", type=int, default=30_000)
    parser.add_argument("--student-batch-size", type=int, default=1024)
    parser.add_argument("--student-learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-envs", type=int, default=128)
    parser.add_argument("--long-envs", type=int, default=16)
    parser.add_argument("--long-steps", type=int, default=1500)
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read required T9 artifact {path}: {exc}") from exc


def _config_from_policy_metadata(artifact):
    stored = artifact.metadata.get("config")
    if not isinstance(stored, dict):
        raise SystemExit("frozen T8 metadata has no config")
    allowed = {field.name for field in fields(ForwardTeacherStudentConfig)}
    return ForwardTeacherStudentConfig(
        **{key: value for key, value in stored.items() if key in allowed}
    )


def _student_speed_gate(
    speed,
    student,
    teacher,
    disturbed_student,
    disturbed_teacher,
    long_student,
    long_teacher,
):
    speed_error_limit = 0.02
    short_ok = (
        student["mean_velocity_error"] <= speed_error_limit
        and student["failure_rate"] <= teacher["failure_rate"] + 0.05
        and student["mean_roll_pitch_rate_rms"]
        <= teacher["mean_roll_pitch_rate_rms"] + 0.12
        and student["mean_abs_velocity_y"] <= teacher["mean_abs_velocity_y"] + 0.02
        and student["mean_abs_yaw_rate"] <= teacher["mean_abs_yaw_rate"] + 0.08
    )
    if speed == 0.0:
        short_ok = short_ok and abs(student["mean_velocity_x"]) <= 0.02
    disturbed_ok = (
        disturbed_student["failure_rate"] <= disturbed_teacher["failure_rate"] + 0.05
        and disturbed_student["mean_post_push_velocity_error"]
        <= disturbed_teacher["mean_post_push_velocity_error"] + 0.04
        and disturbed_student["mean_recovery_time"]
        <= disturbed_teacher["mean_recovery_time"] + 0.50
        and disturbed_student["mean_disk_contacts"]
        <= disturbed_teacher["mean_disk_contacts"] + 0.02
    )
    long_ok = (
        long_student["failure_rate"] == 0.0
        and long_student["disk_contact_environment_rate"] == 0.0
        and long_student["force_saturation_fraction"] < 0.01
        and long_student["mean_absolute_lateral_displacement_m"]
        <= long_teacher["mean_absolute_lateral_displacement_m"] + 0.15
        and long_student["mean_absolute_yaw_change_rad"]
        <= long_teacher["mean_absolute_yaw_change_rad"] + 0.15
    )
    return {
        "accepted": bool(short_ok and disturbed_ok and long_ok),
        "short_horizon_tracking": bool(short_ok),
        "disturbed_retention": bool(disturbed_ok),
        "long_horizon_retention": bool(long_ok),
    }


def _counterfactual_report(jax, jp, params, obs_mean, obs_std, observations, anchors):
    from scripts.train_forward_teacher_student import _normalized_student_apply

    selected = jp.asarray(observations[: min(len(observations), 2048)])
    policy = jax.jit(
        lambda obs: _normalized_student_apply(jp, params, obs, obs_mean, obs_std)
    )
    actions = []
    for speed in anchors:
        counterfactual = selected.at[:, 132:135].set(jp.asarray((speed, 0.0, 0.0)))
        actions.append(np.asarray(jax.device_get(policy(counterfactual))))
    actions = np.stack(actions)
    adjacent = np.sqrt(np.mean(np.square(np.diff(actions, axis=0)), axis=-1))
    endpoint = np.sqrt(np.mean(np.square(actions[-1] - actions[0]), axis=-1))
    return {
        "samples": int(actions.shape[1]),
        "anchors": list(anchors),
        "mean_adjacent_action_rms": [float(value) for value in np.mean(adjacent, axis=1)],
        "mean_stop_to_max_action_rms": float(np.mean(endpoint)),
        "nontrivial_action_response": bool(np.mean(endpoint) >= 0.01),
        "note": "This is paired with closed-loop speed-grid monotonicity; Jacobian magnitude alone is not acceptance evidence.",
    }


def main(argv=None):
    args = parse_args(argv)
    if not 0.0 < args.nominal_fraction < 1.0:
        raise SystemExit("--nominal-fraction must be in (0, 1)")
    teacher_run = args.teacher_run.expanduser().resolve()
    run_config = _read_json(teacher_run / "run_config.json")
    teacher_evaluation = _read_json(teacher_run / "teacher" / "evaluation.json")
    grid_evaluation = _read_json(teacher_run / "teacher" / "grid_evaluation.json")
    if run_config.get("stage") != "T9_FORWARD_COMMAND_TEACHER":
        raise SystemExit("distillation requires a T9 command-grid Teacher")
    if not teacher_evaluation.get("accepted") or not grid_evaluation.get("accepted"):
        raise SystemExit("distillation requires accepted T9 aggregate and speed-grid gates")
    if teacher_evaluation.get("selected_source") != "ppo":
        raise SystemExit("distillation requires a selected PPO Teacher")
    params_path = teacher_run / "teacher" / "params"
    if not params_path.exists():
        raise SystemExit(f"T9 Teacher params are missing: {params_path}")
    anchors = validate_forward_speed_anchors(run_config.get("command_vx_grid", ()))
    specs = tuple(
        IKReferenceSpec(**values)
        for values in run_config.get("resolved_ik_reference_bank", ())
    )
    xml_path = _resolve_xml_path(run_config, args.xml_path)
    base_config = make_t9_config(_config_from_teacher_run(run_config), anchors)
    base_config = replace(
        base_config, max_episode_steps=max(base_config.max_episode_steps, args.long_steps + 1)
    )
    if base_config.student_policy_observation_size != 138:
        raise SystemExit("T9 Student observation contract must be 138")
    reference = build_ik_reference_bank(xml_path, anchors, specs)
    if args.out is None:
        args.out = teacher_run.parent / f"student_t9_vx_grid_bc_seed{args.seed}"
    out = args.out.expanduser().resolve()

    if args.smoke:
        args.dataset_samples = min(args.dataset_samples, 4096)
        args.rollout_envs = min(args.rollout_envs, 16)
        args.rollout_horizon = min(args.rollout_horizon, 64)
        args.student_updates = min(args.student_updates, 20)
        args.student_batch_size = min(args.student_batch_size, 256)
        args.eval_envs = min(args.eval_envs, 16)
        args.long_envs = min(args.long_envs, 4)
        args.long_steps = min(args.long_steps, 64)

    configure_cloud_runtime(mujoco_gl=args.mujoco_gl, verbose=True)
    try:
        import jax
        import jax.numpy as jp
        import optax
        from brax.io import model as model_io
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise SystemExit(f"T9 distillation requires the mjx312 stack: {exc}") from exc
    out.mkdir(parents=True, exist_ok=True)

    teacher_env = make_forward_teacher_student_env(
        "teacher", config=base_config, reference=reference, xml_path=xml_path, seed=args.seed
    )
    networks = make_network_factory(
        run_config.get("teacher_hidden", [256, 256, 128]), "elu"
    )(
        observation_size=teacher_env.observation_size,
        action_size=teacher_env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    teacher_policy = ppo_networks.make_inference_fn(networks)(
        model_io.load_params(params_path), deterministic=True
    )

    nominal_count = int(round(args.dataset_samples * args.nominal_fraction))
    disturbed_count = args.dataset_samples - nominal_count
    nominal_env = make_forward_teacher_student_env(
        "teacher",
        config=replace(base_config, disturbance_enabled=False),
        reference=reference,
        xml_path=xml_path,
        seed=args.seed + 10_000,
    )
    disturbed_env = make_forward_teacher_student_env(
        "teacher",
        config=replace(base_config, disturbance_enabled=True),
        reference=reference,
        xml_path=xml_path,
        seed=args.seed + 20_000,
    )
    nominal_obs, nominal_actions = _collect_teacher_dataset(
        jax, jp, nominal_env, teacher_policy, args.seed + 30_000,
        args.rollout_envs, args.rollout_horizon, nominal_count, "student_policy_obs",
    )
    disturbed_obs, disturbed_actions = _collect_teacher_dataset(
        jax, jp, disturbed_env, teacher_policy, args.seed + 40_000,
        args.rollout_envs, args.rollout_horizon, disturbed_count, "student_policy_obs",
    )
    observations = np.concatenate((nominal_obs, disturbed_obs)).astype(np.float32)
    labels = np.concatenate((nominal_actions, disturbed_actions)).astype(np.float32)
    order = np.random.default_rng(args.seed + 50_000).permutation(len(observations))
    observations, labels = observations[order], labels[order]
    dataset_command_counts = {
        f"{speed:.2f}": int(np.sum(np.isclose(observations[:, 132], speed, atol=1e-6)))
        for speed in anchors
    }
    if any(count == 0 for count in dataset_command_counts.values()):
        raise SystemExit(
            f"T9 dataset does not cover every command anchor: {dataset_command_counts}"
        )
    obs_mean = jp.asarray(np.mean(observations, axis=0).astype(np.float32))
    obs_std = jp.asarray(np.maximum(np.std(observations, axis=0), 1e-3).astype(np.float32))
    params = _student_init(
        jax, jp, jax.random.PRNGKey(args.seed + 60_000),
        [138, *args.student_hidden, base_config.action_size],
    )
    params = _train_student(
        jax, jp, optax, params, observations, labels, obs_mean, obs_std,
        args.student_updates, args.student_batch_size, args.student_learning_rate,
        args.seed + 70_000, "t9_student_bc",
    )
    if args.save_dataset:
        np.savez_compressed(
            out / "student_t9_dataset.npz", observations=observations, actions=labels
        )

    student_policy = jax.jit(
        lambda obs: _normalized_student_apply(jp, params, obs, obs_mean, obs_std)
    )
    speed_reports = {}
    speeds_observed = []
    for index, speed in enumerate(anchors):
        fixed = replace(
            base_config,
            command_vx=speed,
            command_vx_values=(),
            disturbance_enabled=False,
            fixed_reset_phase=0.0,
        )
        student_env = make_forward_teacher_student_env(
            "student", config=fixed, xml_path=xml_path, seed=0
        )
        paired_teacher_env = make_forward_teacher_student_env(
            "teacher", config=fixed, reference=reference, xml_path=xml_path, seed=0
        )
        disturbed_student_env = make_forward_teacher_student_env(
            "student",
            config=replace(fixed, disturbance_enabled=True),
            xml_path=xml_path,
            seed=0,
        )
        disturbed_teacher_env = make_forward_teacher_student_env(
            "teacher",
            config=replace(fixed, disturbance_enabled=True),
            reference=reference,
            xml_path=xml_path,
            seed=0,
        )
        short_seed = args.seed + 80_000 + index
        short = _evaluate_student(
            jax, jp, student_env, params, obs_mean, obs_std,
            short_seed, args.eval_envs, 500,
        )
        paired_teacher_short = _evaluate_teacher(
            jax,
            paired_teacher_env,
            teacher_policy,
            short_seed,
            args.eval_envs,
            500,
        )
        disturbed_short = _evaluate_student(
            jax,
            jp,
            disturbed_student_env,
            params,
            obs_mean,
            obs_std,
            short_seed + 5000,
            args.eval_envs,
            500,
        )
        paired_teacher_disturbed = _evaluate_teacher(
            jax,
            disturbed_teacher_env,
            teacher_policy,
            short_seed + 5000,
            args.eval_envs,
            500,
        )
        long_seed = args.seed + 90_000 + index
        trace = _rollout_policy(
            jax, jp, student_env, "student", teacher_policy, student_policy,
            long_seed, args.long_envs, args.long_steps,
        )
        teacher_trace = _rollout_policy(
            jax,
            jp,
            paired_teacher_env,
            "teacher",
            teacher_policy,
            None,
            long_seed,
            args.long_envs,
            args.long_steps,
        )
        trace.pop("qpos")
        teacher_trace.pop("qpos")
        long_report = summarize_trajectory(
            trace, dt=student_env.dt, torque_limit=fixed.torque_limit,
            ctrl_low=student_env.contract.ctrl_low, ctrl_high=student_env.contract.ctrl_high,
        )
        paired_teacher_long = summarize_trajectory(
            teacher_trace,
            dt=paired_teacher_env.dt,
            torque_limit=fixed.torque_limit,
            ctrl_low=paired_teacher_env.contract.ctrl_low,
            ctrl_high=paired_teacher_env.contract.ctrl_high,
        )
        gate = _student_speed_gate(
            speed,
            short,
            paired_teacher_short,
            disturbed_short,
            paired_teacher_disturbed,
            long_report,
            paired_teacher_long,
        )
        speed_reports[f"{speed:.2f}"] = {
            **gate,
            "command_vx": speed,
            "student_short": short,
            "teacher_short": paired_teacher_short,
            "student_disturbed": disturbed_short,
            "teacher_disturbed": paired_teacher_disturbed,
            "student_long_horizon": long_report,
            "teacher_long_horizon": paired_teacher_long,
        }
        speeds_observed.append(short["mean_velocity_x"])
        print(
            f"stage=t9_student_speed vx={speed:.2f} actual={short['mean_velocity_x']:.4f} "
            f"error={short['mean_velocity_error']:.4f} accepted={gate['accepted']} "
            f"lateral30={long_report['mean_absolute_lateral_displacement_m']:.4f}",
            flush=True,
        )

    monotonic = bool(np.all(np.diff(np.asarray(speeds_observed)) >= -0.005))
    counterfactual = _counterfactual_report(
        jax, jp, params, obs_mean, obs_std, observations, anchors
    )

    t8_path = args.t8_run.expanduser().resolve() / "student_policy_phase_bc_no_previous_action.npz"
    t8_artifact = load_student_policy(t8_path)
    if t8_artifact.metadata.get("stage") != "T8_PHASE_BC_NO_PREVIOUS_ACTION":
        raise SystemExit("--t8-run is not the frozen accepted T8 artifact")
    t8_config = replace(
        _config_from_policy_metadata(t8_artifact),
        command_vx=0.08,
        command_vx_values=(),
        disturbance_enabled=False,
        max_episode_steps=max(args.long_steps + 1, 501),
    )
    retention_seed = args.seed + 100_000
    t8_config = replace(t8_config, fixed_reset_phase=0.0)
    t8_env = make_forward_teacher_student_env(
        "student", config=t8_config, xml_path=xml_path, seed=0
    )
    t8_trace = _rollout_policy(
        jax, jp, t8_env, "student", teacher_policy,
        jax.jit(make_student_policy_jax(t8_artifact)),
        retention_seed, args.long_envs, args.long_steps,
    )
    t8_trace.pop("qpos")
    t8_long = summarize_trajectory(
        t8_trace, dt=t8_env.dt, torque_limit=t8_config.torque_limit,
        ctrl_low=t8_env.contract.ctrl_low, ctrl_high=t8_env.contract.ctrl_high,
    )
    t9_retention_config = replace(
        base_config,
        command_vx=0.08,
        command_vx_values=(),
        disturbance_enabled=False,
        fixed_reset_phase=0.0,
    )
    t9_retention_env = make_forward_teacher_student_env(
        "student", config=t9_retention_config, xml_path=xml_path, seed=0
    )
    t9_retention_trace = _rollout_policy(
        jax,
        jp,
        t9_retention_env,
        "student",
        teacher_policy,
        student_policy,
        retention_seed,
        args.long_envs,
        args.long_steps,
    )
    t9_retention_trace.pop("qpos")
    t9_008 = summarize_trajectory(
        t9_retention_trace,
        dt=t9_retention_env.dt,
        torque_limit=t9_retention_config.torque_limit,
        ctrl_low=t9_retention_env.contract.ctrl_low,
        ctrl_high=t9_retention_env.contract.ctrl_high,
    )
    t8_retained = (
        t9_008["failure_rate"] <= t8_long["failure_rate"]
        and t9_008["mean_forward_displacement_m"]
        >= t8_long["mean_forward_displacement_m"] - 0.15
        and t9_008["mean_absolute_lateral_displacement_m"]
        <= t8_long["mean_absolute_lateral_displacement_m"] + 0.10
        and t9_008["mean_absolute_yaw_change_rad"]
        <= t8_long["mean_absolute_yaw_change_rad"] + 0.10
    )
    accepted = (
        all(report["accepted"] for report in speed_reports.values())
        and monotonic
        and counterfactual["nontrivial_action_response"]
        and t8_retained
    )
    report = {
        "stage": "T9_FORWARD_COMMAND_BC",
        "accepted": bool(accepted),
        "teacher_run": str(teacher_run),
        "t8_run": str(args.t8_run.expanduser().resolve()),
        "anchors": list(anchors),
        "observation_size": 138,
        "dataset_command_counts": dataset_command_counts,
        "speed_reports": speed_reports,
        "closed_loop_speed_monotonic": monotonic,
        "counterfactual": counterfactual,
        "t8_retention_at_0_08": {
            "accepted": bool(t8_retained),
            "t9": t9_008,
            "t8": t8_long,
        },
    }
    metadata = {
        "format": "disk_robot_student_mlp_v1",
        "stage": "T9_FORWARD_COMMAND_BC",
        "xml_path": str(xml_path),
        "observation_size": 138,
        "observation_contract": (
            "four 33-value physical frames without previous action or repeated command, "
            "then current [vx,vy,wz], sin_phase, cos_phase, gait_blend"
        ),
        "previous_action_input": False,
        "current_command_only": True,
        "command_anchors": list(anchors),
        "action_size": base_config.action_size,
        "hidden_layers": args.student_hidden,
        "student_action_scale": list(base_config.student_action_scale),
        "config": asdict(base_config),
        "teacher_run": str(teacher_run),
        "evaluation": report,
    }
    policy_path = out / "student_policy_t9_forward_command_bc.npz"
    _save_student_policy(policy_path, params, obs_mean, obs_std, metadata)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "run_config.json").write_text(
        json.dumps({**vars(args), "out": str(out), "xml_path": str(xml_path)}, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        f"stage=t9_student_acceptance accepted={accepted} monotonic={monotonic} "
        f"counterfactual={counterfactual['nontrivial_action_response']} "
        f"t8_retained={t8_retained} policy={policy_path}",
        flush=True,
    )
    if args.strict and not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
