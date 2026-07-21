from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from disk_robot.ik_reference import build_ik_reference
from disk_robot.student_policy import apply_student_policy_numpy
from disk_robot_mjx.pipeline import configure_cloud_runtime, make_network_factory
from disk_robot_mjx.teacher_student_env import make_forward_teacher_student_env
from scripts.audit_phase_student_failure import _array_stats, _reset_at_deploy_phase
from scripts.dagger_forward_student import (
    _config_from_student_artifact,
    _load_bc_run,
    _validate_bc_teacher_contract,
)
from scripts.distill_forward_student import (
    _config_from_teacher_run,
    _load_accepted_teacher_run,
    _reference_spec_from_teacher_run,
    _resolve_xml_path,
)
from scripts.train_forward_teacher_student import _normalized_student_apply


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose which deployable observation group amplifies a small T5 "
            "Student action error into closed-loop divergence. No policy is trained."
        )
    )
    parser.add_argument("--teacher-run", type=Path)
    parser.add_argument("--student-run", type=Path)
    parser.add_argument(
        "--report-in",
        type=Path,
        default=None,
        help="Print a previously saved long-horizon report without JAX or another rollout.",
    )
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=180_000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--phase-bins", type=int, default=20)
    parser.add_argument(
        "--summary-windows",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 10.0, 20.0, 30.0],
        help="Window edges in seconds for long-horizon action-bias summaries.",
    )
    return parser.parse_args(argv)


def observation_group_indices(
    observation_history=4, frame_size=48, previous_action_input=True
):
    command_start = 45 if previous_action_input else 33
    frame_groups = {
        "imu_and_body_velocity_history": np.concatenate(
            [np.arange(frame * frame_size, frame * frame_size + 9) for frame in range(observation_history)]
        ),
        "joint_position_history": np.concatenate(
            [np.arange(frame * frame_size + 9, frame * frame_size + 21) for frame in range(observation_history)]
        ),
        "joint_velocity_history": np.concatenate(
            [np.arange(frame * frame_size + 21, frame * frame_size + 33) for frame in range(observation_history)]
        ),
        "command_history": np.concatenate(
            [np.arange(frame * frame_size + command_start, (frame + 1) * frame_size) for frame in range(observation_history)]
        ),
    }
    if previous_action_input:
        frame_groups = {
            "latest_previous_action": np.arange(33, 45),
            "previous_action_history": np.concatenate(
                [np.arange(frame * frame_size + 33, frame * frame_size + 45) for frame in range(observation_history)]
            ),
            **frame_groups,
        }
    return {
        **frame_groups,
        "phase_clock": np.arange(
            observation_history * frame_size,
            observation_history * frame_size + 3,
        ),
    }


def _policy_jacobian_numpy(artifact, observation):
    observation = np.asarray(observation, dtype=np.float64)
    obs_mean = np.asarray(artifact.obs_mean, dtype=np.float64)
    obs_std = np.asarray(artifact.obs_std, dtype=np.float64)
    raw_normalized = (observation - obs_mean) / obs_std
    value = np.clip(raw_normalized, -10.0, 10.0)
    input_derivative = np.where(np.abs(raw_normalized) < 10.0, 1.0 / obs_std, 0.0)
    derivative = None
    for index, (weight, bias) in enumerate(artifact.params):
        weight = np.asarray(weight, dtype=np.float64)
        bias = np.asarray(bias, dtype=np.float64)
        preactivation = value @ weight + bias
        derivative = (
            weight * input_derivative[:, None]
            if derivative is None
            else derivative @ weight
        )
        if index < len(artifact.params) - 1:
            activation_derivative = np.where(preactivation > 0.0, 1.0, np.exp(preactivation))
            derivative *= activation_derivative[None, :]
            value = np.where(preactivation > 0.0, preactivation, np.expm1(preactivation))
        else:
            value = np.tanh(preactivation)
            derivative *= (1.0 - np.square(value))[None, :]
    return derivative


def _collect_trace(
    jax,
    jp,
    env,
    teacher_policy,
    student_params,
    obs_mean,
    obs_std,
    seed,
    env_count,
    steps,
):
    reset_batch = jax.jit(jax.vmap(lambda key: _reset_at_deploy_phase(env, jp, key)))
    step_batch = jax.jit(jax.vmap(env.step))
    label_batch = jax.jit(jax.vmap(env.teacher_action_to_student_action))
    student_policy = jax.jit(
        lambda obs: _normalized_student_apply(jp, student_params, obs, obs_mean, obs_std)
    )
    state = reset_batch(jax.random.split(jax.random.PRNGKey(seed), env_count))

    def rollout_step(carry, _):
        oracle_state, student_state, policy_key = carry
        policy_key, oracle_key, correction_key = jax.random.split(policy_key, 3)
        oracle_residual, _ = teacher_policy(oracle_state.info["teacher_obs"], oracle_key)
        correction_residual, _ = teacher_policy(
            student_state.info["teacher_obs"], correction_key
        )
        oracle_label = label_batch(oracle_state, oracle_residual)
        teacher_correction = label_batch(student_state, correction_residual)
        student_on_oracle = student_policy(oracle_state.obs)
        student_action = student_policy(student_state.obs)
        next_oracle = step_batch(oracle_state, oracle_label)
        next_student = step_batch(student_state, student_action)
        q_error = (
            next_student.pipeline_state.qpos[:, env.qpos_indices]
            - next_oracle.pipeline_state.qpos[:, env.qpos_indices]
        )
        return (next_oracle, next_student, policy_key), (
            oracle_state.obs,
            student_state.obs,
            oracle_label,
            teacher_correction,
            student_on_oracle,
            student_action,
            q_error,
        )

    (_, _, _), values = jax.lax.scan(
        rollout_step,
        (state, state, jax.random.PRNGKey(seed + 1)),
        (),
        length=steps,
    )
    return tuple(np.asarray(jax.device_get(value)) for value in values)


def _rms_per_step(values):
    return np.sqrt(np.mean(np.square(values), axis=-1)).mean(axis=1)


def _group_attribution(artifact, oracle_obs, student_obs, oracle_policy_action, groups):
    flat_student_obs = student_obs.reshape(-1, student_obs.shape[-1])
    flat_oracle_obs = oracle_obs.reshape(-1, oracle_obs.shape[-1])
    flat_oracle_action = oracle_policy_action.reshape(-1, oracle_policy_action.shape[-1])
    baseline_action = apply_student_policy_numpy(artifact, flat_student_obs)
    baseline_shift = baseline_action - flat_oracle_action
    baseline_rms = np.sqrt(np.mean(np.square(baseline_shift), axis=1))
    reports = {}
    for name, indices in groups.items():
        hybrid = flat_student_obs.copy()
        hybrid[:, indices] = flat_oracle_obs[:, indices]
        hybrid_action = apply_student_policy_numpy(artifact, hybrid)
        hybrid_shift = hybrid_action - flat_oracle_action
        hybrid_rms = np.sqrt(np.mean(np.square(hybrid_shift), axis=1))
        recovered = np.where(
            baseline_rms > 1e-8,
            1.0 - hybrid_rms / np.maximum(baseline_rms, 1e-8),
            0.0,
        )
        normalized_delta = (
            flat_student_obs[:, indices] - flat_oracle_obs[:, indices]
        ) / np.asarray(artifact.obs_std)[indices]
        reports[name] = {
            "indices": [int(value) for value in indices],
            "normalized_observation_delta": _array_stats(normalized_delta),
            "hybrid_policy_shift": _array_stats(hybrid_shift),
            "policy_shift_recovered_fraction": _array_stats(recovered),
            "mean_recovered_fraction_steps_1_to_5": float(
                np.mean(recovered.reshape(student_obs.shape[:2])[1:6])
            ),
        }
    return reports


def _jacobian_audit(artifact, observations, groups, max_steps=6):
    selected = observations[:max_steps].reshape(-1, observations.shape[-1])
    reports = {
        name: {"spectral": [], "frobenius": []} for name in groups
    }
    for observation in selected:
        derivative = _policy_jacobian_numpy(artifact, observation)
        for name, indices in groups.items():
            group_derivative = derivative[indices, :]
            singular_values = np.linalg.svd(group_derivative, compute_uv=False)
            reports[name]["spectral"].append(float(singular_values[0]))
            reports[name]["frobenius"].append(float(np.linalg.norm(group_derivative)))
    return {
        name: {
            "samples": len(values["spectral"]),
            "spectral_gain": _array_stats(values["spectral"]),
            "frobenius_gain": _array_stats(values["frobenius"]),
        }
        for name, values in reports.items()
    }


def _long_horizon_bias_audit(
    oracle_obs,
    oracle_label,
    teacher_correction,
    student_on_oracle,
    student_action,
    action_scale,
    dt,
    phase_bins,
    window_edges,
):
    """Separates fixed approximation bias from closed-loop and Teacher-label drift."""

    action_scale = np.asarray(action_scale, dtype=np.float64)
    oracle_error = np.asarray(student_on_oracle) - np.asarray(oracle_label)
    closed_error = np.asarray(student_action) - np.asarray(teacher_correction)
    label_drift = np.asarray(teacher_correction) - np.asarray(oracle_label)
    physical_oracle_error = oracle_error * action_scale
    physical_closed_error = closed_error * action_scale
    phase = np.mod(
        np.arctan2(oracle_obs[..., -3], oracle_obs[..., -2]) / (2.0 * np.pi),
        1.0,
    )

    per_joint = []
    from disk_robot.model_contract import JOINT_NAMES

    for index, name in enumerate(JOINT_NAMES):
        per_joint.append(
            {
                "joint": name,
                "oracle_manifold_action_error": _array_stats(oracle_error[..., index]),
                "closed_loop_action_error": _array_stats(closed_error[..., index]),
                "teacher_label_drift": _array_stats(label_drift[..., index]),
                "oracle_manifold_target_error_rad": _array_stats(
                    physical_oracle_error[..., index]
                ),
                "closed_loop_target_error_rad": _array_stats(
                    physical_closed_error[..., index]
                ),
            }
        )

    bins = []
    bin_index = np.minimum((phase * phase_bins).astype(np.int32), phase_bins - 1)
    for index in range(phase_bins):
        selected = bin_index == index
        bins.append(
            {
                "bin": index,
                "phase_start": float(index / phase_bins),
                "phase_end": float((index + 1) / phase_bins),
                "samples": int(np.sum(selected)),
                "oracle_manifold_action_error": _array_stats(oracle_error[selected]),
                "closed_loop_action_error": _array_stats(closed_error[selected]),
                "teacher_label_drift": _array_stats(label_drift[selected]),
                "closed_loop_mean_target_error_rad_by_joint": [
                    float(value)
                    for value in np.mean(physical_closed_error[selected], axis=0)
                ]
                if np.any(selected)
                else [0.0] * len(JOINT_NAMES),
            }
        )

    horizon_seconds = oracle_error.shape[0] * dt
    edges = sorted(
        {
            min(max(float(value), 0.0), horizon_seconds)
            for value in window_edges
        }
    )
    if not edges or edges[0] > 0.0:
        edges.insert(0, 0.0)
    if edges[-1] < horizon_seconds:
        edges.append(horizon_seconds)
    windows = []
    for start_s, end_s in zip(edges[:-1], edges[1:]):
        start = min(int(round(start_s / dt)), oracle_error.shape[0])
        end = min(int(round(end_s / dt)), oracle_error.shape[0])
        if start >= end:
            continue
        mean_by_joint = np.mean(physical_closed_error[start:end], axis=(0, 1))
        largest = int(np.argmax(np.abs(mean_by_joint)))
        windows.append(
            {
                "start_s": float(start * dt),
                "end_s": float(end * dt),
                "oracle_manifold_action_error": _array_stats(oracle_error[start:end]),
                "closed_loop_action_error": _array_stats(closed_error[start:end]),
                "teacher_label_drift": _array_stats(label_drift[start:end]),
                "closed_loop_target_error_rad": _array_stats(
                    physical_closed_error[start:end]
                ),
                "closed_loop_mean_target_error_rad_by_joint": [
                    float(value) for value in mean_by_joint
                ],
                "largest_mean_bias_joint": JOINT_NAMES[largest],
                "largest_mean_bias_rad": float(mean_by_joint[largest]),
            }
        )

    right_indices = np.asarray((0, 1, 2, 6, 7, 8))
    left_indices = np.asarray((3, 4, 5, 9, 10, 11))
    return {
        "dt": float(dt),
        "duration_s": float(horizon_seconds),
        "oracle_manifold_action_error": _array_stats(oracle_error),
        "closed_loop_action_error": _array_stats(closed_error),
        "teacher_label_drift": _array_stats(label_drift),
        "per_joint": per_joint,
        "phase_bins": bins,
        "temporal_windows": windows,
        "raw_coordinate_side_mean_target_error_rad": {
            "right": float(np.mean(physical_closed_error[..., right_indices])),
            "left": float(np.mean(physical_closed_error[..., left_indices])),
            "right_minus_left": float(
                np.mean(physical_closed_error[..., right_indices])
                - np.mean(physical_closed_error[..., left_indices])
            ),
            "note": "Raw joint coordinates have mirrored axes; this is an asymmetry flag, not a yaw-moment estimate.",
        },
    }


def _print_bias_windows(report):
    bias = report.get("long_horizon_bias")
    if not isinstance(bias, dict) or not isinstance(bias.get("temporal_windows"), list):
        raise SystemExit("saved report has no long_horizon_bias.temporal_windows")
    for window in bias["temporal_windows"]:
        print(
            "stage=feedback_bias_window "
            f"start_s={window['start_s']:.1f} end_s={window['end_s']:.1f} "
            f"oracle_rmse={window['oracle_manifold_action_error']['rmse']:.5f} "
            f"closed_rmse={window['closed_loop_action_error']['rmse']:.5f} "
            f"label_drift={window['teacher_label_drift']['rmse']:.5f} "
            f"largest_joint={window['largest_mean_bias_joint']} "
            f"bias_rad={window['largest_mean_bias_rad']:+.6f}",
            flush=True,
        )
def main(argv=None):
    args = parse_args(argv)
    if args.report_in is not None:
        try:
            saved_report = json.loads(
                args.report_in.expanduser().resolve().read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise SystemExit(f"could not read saved feedback report: {exc}") from exc
        _print_bias_windows(saved_report)
        print(
            f"stage=feedback_saved_report_complete report={args.report_in.expanduser().resolve()}",
            flush=True,
        )
        return
    if args.teacher_run is None or args.student_run is None:
        raise SystemExit("--teacher-run and --student-run are required unless --report-in is used")
    if args.envs < 2 or args.steps < 3 or args.phase_bins < 1:
        raise SystemExit("--envs must be at least 2, --steps at least 3, and --phase-bins positive")
    if len(args.summary_windows) < 2:
        raise SystemExit("--summary-windows requires at least two edges")
    teacher_run, run_config, teacher_evaluation, _, params_path = (
        _load_accepted_teacher_run(args.teacher_run)
    )
    teacher_config = _config_from_teacher_run(run_config)
    student_run, student_path, artifact, _, _, _ = _load_bc_run(args.student_run)
    config = _config_from_student_artifact(artifact, teacher_config)
    _validate_bc_teacher_contract(
        artifact, teacher_run, teacher_evaluation, config, teacher_config
    )
    if artifact.metadata.get("stage") not in {
        "T5_PHASE_BC",
        "T8_PHASE_BC_NO_PREVIOUS_ACTION",
    }:
        raise SystemExit("Feedback diagnosis requires a T5 or T8 phase BC policy")
    reference_spec = _reference_spec_from_teacher_run(run_config)
    xml_path = _resolve_xml_path(run_config, args.xml_path)
    default_report_name = (
        "feedback_long_horizon_diagnosis.json" if args.steps > 100 else "feedback_diagnosis.json"
    )
    default_trace_name = (
        "feedback_long_horizon_trace.npz" if args.steps > 100 else "feedback_diagnosis_trace.npz"
    )
    output_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else student_run / default_report_name
    )
    trace_path = (
        args.trace_out.expanduser().resolve()
        if args.trace_out is not None
        else student_run / default_trace_name
    )

    configure_cloud_runtime(mujoco_gl=args.mujoco_gl, verbose=True)
    try:
        import jax
        import jax.numpy as jp
        from brax.io import model as model_io
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise SystemExit(f"Feedback diagnosis requires the mjx312 stack: {exc}") from exc

    reference = build_ik_reference(xml_path, reference_spec)
    env = make_forward_teacher_student_env(
        "dagger",
        config=replace(
            config,
            disturbance_enabled=False,
            max_episode_steps=max(config.max_episode_steps, args.steps + 1),
        ),
        reference=reference,
        xml_path=xml_path,
        seed=args.seed,
    )
    teacher_networks = make_network_factory(
        run_config.get("teacher_hidden", [256, 256, 128]), "elu"
    )(
        observation_size=env.config.teacher_observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    teacher_policy = ppo_networks.make_inference_fn(teacher_networks)(
        model_io.load_params(params_path), deterministic=True
    )
    student_params = tuple(
        (jp.asarray(weight), jp.asarray(bias)) for weight, bias in artifact.params
    )
    values = _collect_trace(
        jax,
        jp,
        env,
        teacher_policy,
        student_params,
        jp.asarray(artifact.obs_mean),
        jp.asarray(artifact.obs_std),
        args.seed,
        args.envs,
        args.steps,
    )
    (
        oracle_obs,
        student_obs,
        oracle_label,
        teacher_correction,
        student_on_oracle,
        student_action,
        q_error,
    ) = values
    oracle_manifold_error = _rms_per_step(student_on_oracle - oracle_label)
    closed_loop_error = _rms_per_step(student_action - teacher_correction)
    policy_shift = _rms_per_step(student_action - student_on_oracle)
    label_drift = _rms_per_step(teacher_correction - oracle_label)
    q_rmse = _rms_per_step(q_error)
    groups = observation_group_indices(
        config.observation_history,
        config.student_policy_frame_size,
        config.student_previous_action_input,
    )
    attribution = _group_attribution(
        artifact, oracle_obs, student_obs, student_on_oracle, groups
    )
    jacobian = _jacobian_audit(artifact, student_obs, groups)
    long_horizon_bias = _long_horizon_bias_audit(
        oracle_obs,
        oracle_label,
        teacher_correction,
        student_on_oracle,
        student_action,
        config.student_action_scale,
        env.dt,
        args.phase_bins,
        args.summary_windows,
    )

    np.savez_compressed(
        trace_path,
        oracle_observations=oracle_obs,
        student_observations=student_obs,
        oracle_labels=oracle_label,
        teacher_corrections=teacher_correction,
        student_on_oracle=student_on_oracle,
        student_actions=student_action,
        joint_position_error=q_error,
    )
    report = {
        "stage": (
            "T8_LONG_HORIZON_BIAS_DIAGNOSIS"
            if artifact.metadata.get("stage") == "T8_PHASE_BC_NO_PREVIOUS_ACTION"
            and args.steps > 100
            else "T7_FEEDBACK_DIAGNOSIS"
        ),
        "teacher_run": str(teacher_run),
        "student_policy": str(student_path),
        "envs": args.envs,
        "steps": args.steps,
        "previous_action_input": config.student_previous_action_input,
        "per_step": [
            {
                "step": index,
                "oracle_manifold_error": float(oracle_manifold_error[index]),
                "closed_loop_error": float(closed_loop_error[index]),
                "policy_shift": float(policy_shift[index]),
                "teacher_label_drift": float(label_drift[index]),
                "joint_position_rmse_rad": float(q_rmse[index]),
            }
            for index in range(args.steps)
        ],
        "group_counterfactual": attribution,
        "local_jacobian": jacobian,
        "long_horizon_bias": long_horizon_bias,
        "trace": str(trace_path),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for index in range(min(args.steps, 10)):
        print(
            "stage=feedback_step "
            f"step={index:02d} oracle_error={oracle_manifold_error[index]:.5f} "
            f"closed_error={closed_loop_error[index]:.5f} "
            f"policy_shift={policy_shift[index]:.5f} "
            f"label_drift={label_drift[index]:.5f} q_rmse={q_rmse[index]:.5f}",
            flush=True,
        )
    ranked = sorted(
        attribution,
        key=lambda name: attribution[name]["mean_recovered_fraction_steps_1_to_5"],
        reverse=True,
    )
    for name in ranked:
        print(
            "stage=feedback_group "
            f"group={name} "
            f"recovered={attribution[name]['mean_recovered_fraction_steps_1_to_5']:+.3f} "
            f"obs_delta={attribution[name]['normalized_observation_delta']['rmse']:.4f} "
            f"jacobian_spectral={jacobian[name]['spectral_gain']['mean']:.3f}",
            flush=True,
        )
    _print_bias_windows(report)
    print(
        f"stage=feedback_diagnosis_complete report={output_path} trace={trace_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
