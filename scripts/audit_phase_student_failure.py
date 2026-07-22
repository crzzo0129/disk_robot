from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from disk_robot.ik_reference import build_ik_reference
from disk_robot.model_contract import JOINT_NAMES
from disk_robot.student_policy import apply_student_policy_numpy
from disk_robot_mjx.pipeline import configure_cloud_runtime, make_network_factory
from disk_robot_mjx.teacher_student_env import make_forward_teacher_student_env
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
            "Audit why a phase-conditioned Student fails despite low offline BC loss. "
            "This script does not train or modify a policy."
        )
    )
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--student-run", type=Path, required=True)
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=130_000)
    parser.add_argument("--phase-bins", type=int, default=20)
    parser.add_argument("--nearest-neighbor-samples", type=int, default=2_048)
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.002, 0.005, 0.01],
        help="Direct Student-action perturbation RMS values.",
    )
    parser.add_argument("--q-divergence-rmse", type=float, default=0.01)
    parser.add_argument("--torso-divergence-m", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _array_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "mean_abs": 0.0,
            "rmse": 0.0,
            "p95_abs": 0.0,
            "maximum_abs": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "p95_abs": float(np.percentile(np.abs(values), 95.0)),
        "maximum_abs": float(np.max(np.abs(values))),
    }


def _phase_from_observations(observations):
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] < 3:
        raise ValueError("phase-conditioned observations must be a two-dimensional array")
    return np.mod(np.arctan2(observations[:, -3], observations[:, -2]) / (2.0 * np.pi), 1.0)


def _offline_error_audit(artifact, observations, labels, action_scale, phase_bins):
    predictions = apply_student_policy_numpy(artifact, observations)
    errors = predictions - labels
    target_errors = errors * np.asarray(action_scale, dtype=np.float32)
    phase = _phase_from_observations(observations)
    per_joint = []
    for index, name in enumerate(JOINT_NAMES):
        per_joint.append(
            {
                "joint": name,
                "action_error": _array_stats(errors[:, index]),
                "target_error_rad": _array_stats(target_errors[:, index]),
            }
        )
    phase_reports = []
    bin_index = np.minimum((phase * phase_bins).astype(np.int32), phase_bins - 1)
    for index in range(phase_bins):
        selected = bin_index == index
        bin_errors = errors[selected]
        bin_target_errors = target_errors[selected]
        phase_reports.append(
            {
                "bin": index,
                "phase_start": float(index / phase_bins),
                "phase_end": float((index + 1) / phase_bins),
                "samples": int(np.sum(selected)),
                "action_error": _array_stats(bin_errors),
                "target_error_rad": _array_stats(bin_target_errors),
                "largest_bias_joint": (
                    JOINT_NAMES[int(np.argmax(np.abs(np.mean(bin_errors, axis=0))))]
                    if len(bin_errors)
                    else None
                ),
            }
        )
    joint_rmse = np.asarray([entry["action_error"]["rmse"] for entry in per_joint])
    joint_bias = np.asarray([entry["action_error"]["mean"] for entry in per_joint])
    return {
        "samples": int(len(observations)),
        "action_error": _array_stats(errors),
        "target_error_rad": _array_stats(target_errors),
        "largest_rmse_joint": JOINT_NAMES[int(np.argmax(joint_rmse))],
        "largest_bias_joint": JOINT_NAMES[int(np.argmax(np.abs(joint_bias)))],
        "per_joint": per_joint,
        "phase_bins": phase_reports,
    }


def _first_crossing(values, threshold):
    values = np.asarray(values)
    crossed = values > threshold
    first = np.argmax(crossed, axis=0)
    return np.where(np.any(crossed, axis=0), first, -1)


def _crossing_summary(first_steps, dt):
    first_steps = np.asarray(first_steps)
    valid = first_steps >= 0
    if not np.any(valid):
        return {
            "coverage": 0.0,
            "median_step": None,
            "median_seconds": None,
            "minimum_step": None,
        }
    return {
        "coverage": float(np.mean(valid)),
        "median_step": float(np.median(first_steps[valid])),
        "median_seconds": float(np.median(first_steps[valid]) * dt),
        "minimum_step": int(np.min(first_steps[valid])),
    }


def _nearest_neighbor_audit(
    observations,
    teacher_labels,
    student_actions,
    env_ids,
    obs_mean,
    obs_std,
    sample_count,
    seed,
):
    observations = np.asarray(observations, dtype=np.float32)
    teacher_labels = np.asarray(teacher_labels, dtype=np.float32)
    student_actions = np.asarray(student_actions, dtype=np.float32)
    env_ids = np.asarray(env_ids)
    count = min(int(sample_count), len(observations))
    if count < 2:
        return {"samples": count}
    indices = np.random.default_rng(seed).choice(len(observations), count, replace=False)
    normalized = np.clip(
        (observations[indices] - np.asarray(obs_mean)) / np.asarray(obs_std),
        -10.0,
        10.0,
    ).astype(np.float32)
    norms = np.sum(np.square(normalized), axis=1)
    squared_distance = np.maximum(
        norms[:, None] + norms[None, :] - 2.0 * (normalized @ normalized.T),
        0.0,
    )
    same_env = env_ids[indices, None] == env_ids[None, indices]
    squared_distance[same_env] = np.inf
    nearest = np.argmin(squared_distance, axis=1)
    valid = np.isfinite(squared_distance[np.arange(count), nearest])
    selected = np.arange(count)[valid]
    nearest = nearest[valid]
    obs_rms = np.sqrt(
        squared_distance[selected, nearest] / max(1, normalized.shape[1])
    )
    teacher_rms = np.sqrt(
        np.mean(
            np.square(
                teacher_labels[indices[selected]] - teacher_labels[indices[nearest]]
            ),
            axis=1,
        )
    )
    student_rms = np.sqrt(
        np.mean(
            np.square(
                student_actions[indices[selected]] - student_actions[indices[nearest]]
            ),
            axis=1,
        )
    )
    excess = teacher_rms - student_rms
    return {
        "samples": int(len(selected)),
        "cross_environment_only": True,
        "nearest_observation_rms": _array_stats(obs_rms),
        "teacher_label_disagreement_rms": _array_stats(teacher_rms),
        "student_action_disagreement_rms": _array_stats(student_rms),
        "teacher_disagreement_excess": _array_stats(excess),
        "fraction_teacher_disagreement_gt_student_by_0_01": float(
            np.mean(excess > 0.01)
        ),
    }


def _reset_at_deploy_phase(env, jp, rng):
    state = env.reset(rng)
    phase = jp.array(0.0)
    gait_blend = jp.array(0.0)
    command = state.info.get("command", env.command)
    ik_target = env._blended_ik_target(phase, gait_blend, command[0])
    contacts = env._foot_contacts(state.pipeline_state)
    student_policy_obs = env._student_policy_obs(
        state.info["student_obs"], phase, gait_blend, command
    )
    teacher_obs = env._teacher_obs(
        state.pipeline_state,
        state.info["student_obs"],
        phase,
        ik_target,
        contacts,
        state.info["previous_residual"],
        gait_blend,
        state.info["last_push"],
        state.info["motor_strength"],
        state.info["control_delay"],
    )
    info = {
        **state.info,
        "phase": phase,
        "gait_blend": gait_blend,
        "student_policy_obs": student_policy_obs,
        "teacher_obs": teacher_obs,
        "ik_target": ik_target,
    }
    return state.replace(obs=student_policy_obs, info=info)


def _summarize_noisy_rollout(values, env, perturbation_rms):
    reward, vx, roll_pitch, failed, alive = values
    denominator = max(float(np.sum(alive)), 1.0)
    mean_vx = float(np.sum(vx * alive) / denominator)
    return {
        "perturbation_action_rms": float(perturbation_rms),
        "perturbation_target_rms_rad": float(
            perturbation_rms * np.sqrt(np.mean(np.square(env.config.student_action_scale)))
        ),
        "reward_per_step": float(np.sum(reward * alive) / denominator),
        "mean_velocity_x": mean_vx,
        "mean_velocity_error": abs(mean_vx - env.config.command_vx),
        "mean_roll_pitch_rate_rms": float(
            np.sum(roll_pitch * alive) / denominator
        ),
        "failure_rate": float(np.mean(np.max(failed, axis=0))),
        "mean_alive_steps": float(np.mean(np.sum(alive, axis=0))),
    }


def _evaluate_noisy_oracle(
    jax,
    jp,
    env,
    teacher_policy,
    seed,
    env_count,
    horizon,
    magnitude,
    mode,
):
    reset_batch = jax.jit(jax.vmap(lambda key: _reset_at_deploy_phase(env, jp, key)))
    step_batch = jax.jit(jax.vmap(env.step))
    label_batch = jax.jit(jax.vmap(env.teacher_action_to_student_action))
    state = reset_batch(jax.random.split(jax.random.PRNGKey(seed), env_count))
    bias = jax.random.normal(
        jax.random.PRNGKey(seed + 10), (env_count, env.action_size)
    )
    bias = bias / jp.maximum(
        jp.sqrt(jp.mean(jp.square(bias), axis=1, keepdims=True)), 1e-6
    )

    def eval_step(carry, _):
        current_state, policy_key, noise_key = carry
        policy_key, action_key = jax.random.split(policy_key)
        noise_key, step_noise_key = jax.random.split(noise_key)
        residual_action, _ = teacher_policy(
            current_state.info["teacher_obs"], action_key
        )
        direct_action = label_batch(current_state, residual_action)
        if mode == "bias":
            unit_noise = bias
        else:
            unit_noise = jax.random.normal(
                step_noise_key, (env_count, env.action_size)
            )
            unit_noise = unit_noise / jp.maximum(
                jp.sqrt(jp.mean(jp.square(unit_noise), axis=1, keepdims=True)),
                1e-6,
            )
        perturbed_action = jp.clip(direct_action + magnitude * unit_noise, -1.0, 1.0)
        next_state = step_batch(current_state, perturbed_action)
        alive = 1.0 - current_state.done
        return (next_state, policy_key, noise_key), (
            next_state.reward,
            next_state.metrics["velocity_x"],
            next_state.metrics["roll_pitch_rate_rms"],
            next_state.metrics["failed"],
            alive,
        )

    (_, _, _), values = jax.lax.scan(
        eval_step,
        (
            state,
            jax.random.PRNGKey(seed + 20),
            jax.random.PRNGKey(seed + 30),
        ),
        (),
        length=horizon,
    )
    arrays = tuple(np.asarray(jax.device_get(value)) for value in values)
    return _summarize_noisy_rollout(arrays, env, magnitude)


def _paired_divergence_rollout(
    jax,
    jp,
    env,
    teacher_policy,
    student_params,
    obs_mean,
    obs_std,
    seed,
    env_count,
    horizon,
):
    reset_batch = jax.jit(jax.vmap(lambda key: _reset_at_deploy_phase(env, jp, key)))
    step_batch = jax.jit(jax.vmap(env.step))
    label_batch = jax.jit(jax.vmap(env.teacher_action_to_student_action))
    student_policy = jax.jit(
        lambda obs: _normalized_student_apply(
            jp, student_params, obs, obs_mean, obs_std
        )
    )
    state = reset_batch(jax.random.split(jax.random.PRNGKey(seed), env_count))

    def rollout_step(carry, _):
        oracle_state, student_state, policy_key = carry
        policy_key, oracle_key, correction_key = jax.random.split(policy_key, 3)
        oracle_residual, _ = teacher_policy(
            oracle_state.info["teacher_obs"], oracle_key
        )
        correction_residual, _ = teacher_policy(
            student_state.info["teacher_obs"], correction_key
        )
        oracle_label = label_batch(oracle_state, oracle_residual)
        teacher_correction = label_batch(student_state, correction_residual)
        student_action = student_policy(student_state.obs)
        next_oracle = step_batch(oracle_state, oracle_label)
        next_student = step_batch(student_state, student_action)
        q_error = (
            next_student.pipeline_state.qpos[:, env.qpos_indices]
            - next_oracle.pipeline_state.qpos[:, env.qpos_indices]
        )
        qvel_error = (
            next_student.pipeline_state.qvel[:, env.dof_indices]
            - next_oracle.pipeline_state.qvel[:, env.dof_indices]
        )
        torso_error = (
            next_student.pipeline_state.xpos[:, env.torso_body_id]
            - next_oracle.pipeline_state.xpos[:, env.torso_body_id]
        )
        alive = (1.0 - oracle_state.done) * (1.0 - student_state.done)
        return (next_oracle, next_student, policy_key), (
            student_state.obs,
            student_state.info["phase"],
            student_action,
            teacher_correction,
            oracle_label,
            q_error,
            qvel_error,
            torso_error,
            next_oracle.metrics["velocity_x"],
            next_student.metrics["velocity_x"],
            next_student.metrics["roll_pitch_rate_rms"],
            next_student.metrics["failed"],
            alive,
        )

    (_, _, _), values = jax.lax.scan(
        rollout_step,
        (state, state, jax.random.PRNGKey(seed + 1)),
        (),
        length=horizon,
    )
    return tuple(np.asarray(jax.device_get(value)) for value in values)


def _divergence_audit(
    values,
    env,
    obs_mean,
    obs_std,
    q_threshold,
    torso_threshold,
    nearest_neighbor_samples,
    phase_bins,
    seed,
):
    (
        observations,
        phase,
        student_actions,
        teacher_corrections,
        oracle_labels,
        q_error,
        qvel_error,
        torso_error,
        oracle_vx,
        student_vx,
        student_roll_pitch,
        student_failed,
        alive,
    ) = values
    q_rmse = np.sqrt(np.mean(np.square(q_error), axis=-1))
    qvel_rmse = np.sqrt(np.mean(np.square(qvel_error), axis=-1))
    torso_distance = np.linalg.norm(torso_error, axis=-1)
    correction_error = teacher_corrections - student_actions
    correction_rmse = np.sqrt(np.mean(np.square(correction_error), axis=-1))
    label_drift = teacher_corrections - oracle_labels
    label_drift_rmse = np.sqrt(np.mean(np.square(label_drift), axis=-1))
    q_first = _first_crossing(q_rmse, q_threshold)
    torso_first = _first_crossing(torso_distance, torso_threshold)
    denominator = max(float(np.sum(alive)), 1.0)
    env_ids = np.broadcast_to(
        np.arange(observations.shape[1])[None, :], observations.shape[:2]
    ).reshape(-1)
    nearest = _nearest_neighbor_audit(
        observations.reshape(-1, observations.shape[-1]),
        teacher_corrections.reshape(-1, teacher_corrections.shape[-1]),
        student_actions.reshape(-1, student_actions.shape[-1]),
        env_ids,
        obs_mean,
        obs_std,
        nearest_neighbor_samples,
        seed,
    )
    joint_correction = []
    for index, name in enumerate(JOINT_NAMES):
        joint_correction.append(
            {
                "joint": name,
                "student_vs_teacher_action": _array_stats(
                    correction_error[..., index]
                ),
                "teacher_label_drift": _array_stats(label_drift[..., index]),
                "paired_joint_position_error_rad": _array_stats(
                    q_error[..., index]
                ),
            }
        )
    phase_correction = []
    flat_phase = phase.reshape(-1)
    flat_correction = correction_error.reshape(-1, correction_error.shape[-1])
    flat_label_drift = label_drift.reshape(-1, label_drift.shape[-1])
    phase_index = np.minimum(
        (np.mod(flat_phase, 1.0) * phase_bins).astype(np.int32), phase_bins - 1
    )
    for index in range(phase_bins):
        selected = phase_index == index
        phase_correction.append(
            {
                "bin": index,
                "phase_start": float(index / phase_bins),
                "phase_end": float((index + 1) / phase_bins),
                "samples": int(np.sum(selected)),
                "student_teacher_action_error": _array_stats(
                    flat_correction[selected]
                ),
                "teacher_label_drift": _array_stats(flat_label_drift[selected]),
            }
        )
    window_edges = (0, 10, 25, 50, 100, observations.shape[0])
    temporal_windows = []
    for start, end in zip(window_edges[:-1], window_edges[1:]):
        end = min(end, observations.shape[0])
        if start >= end:
            continue
        temporal_windows.append(
            {
                "start_step": int(start),
                "end_step": int(end),
                "student_teacher_action_error": _array_stats(
                    correction_error[start:end]
                ),
                "teacher_label_drift": _array_stats(label_drift[start:end]),
                "paired_joint_position_error_rad": _array_stats(q_error[start:end]),
                "paired_torso_distance_m": _array_stats(torso_distance[start:end]),
            }
        )
    mean_trace = {
        "q_rmse_rad": np.mean(q_rmse, axis=1),
        "qvel_rmse_rad_s": np.mean(qvel_rmse, axis=1),
        "torso_distance_m": np.mean(torso_distance, axis=1),
        "student_teacher_action_rmse": np.mean(correction_rmse, axis=1),
        "teacher_label_drift_rmse": np.mean(label_drift_rmse, axis=1),
        "oracle_vx": np.mean(oracle_vx, axis=1),
        "student_vx": np.mean(student_vx, axis=1),
        "student_roll_pitch_rate_rms": np.mean(student_roll_pitch, axis=1),
    }
    return {
        "q_divergence_threshold_rad": float(q_threshold),
        "torso_divergence_threshold_m": float(torso_threshold),
        "first_q_divergence": _crossing_summary(q_first, env.dt),
        "first_torso_divergence": _crossing_summary(torso_first, env.dt),
        "student_teacher_action_error": _array_stats(correction_error),
        "teacher_label_drift_between_branches": _array_stats(label_drift),
        "paired_joint_position_error_rad": _array_stats(q_error),
        "paired_joint_velocity_error_rad_s": _array_stats(qvel_error),
        "paired_torso_distance_m": _array_stats(torso_distance),
        "oracle_mean_velocity_x": float(np.sum(oracle_vx * alive) / denominator),
        "student_mean_velocity_x": float(np.sum(student_vx * alive) / denominator),
        "student_mean_roll_pitch_rate_rms": float(
            np.sum(student_roll_pitch * alive) / denominator
        ),
        "student_failure_rate": float(np.mean(np.max(student_failed, axis=0))),
        "per_joint": joint_correction,
        "phase_bins": phase_correction,
        "temporal_windows": temporal_windows,
        "nearest_neighbor_identifiability": nearest,
        "mean_trace": {
            key: [float(value) for value in values]
            for key, values in mean_trace.items()
        },
    }


def main(argv=None):
    args = parse_args(argv)
    if min(args.envs, args.steps, args.phase_bins) < 1:
        raise SystemExit("--envs, --steps, and --phase-bins must be positive")
    if args.nearest_neighbor_samples < 2:
        raise SystemExit("--nearest-neighbor-samples must be at least 2")
    if any(level < 0.0 for level in args.noise_levels):
        raise SystemExit("--noise-levels must be non-negative")
    if args.smoke:
        args.envs = min(args.envs, 8)
        args.steps = min(args.steps, 128)
        args.nearest_neighbor_samples = min(args.nearest_neighbor_samples, 512)
        args.noise_levels = [0.0, 0.002]

    teacher_run, run_config, teacher_evaluation, _, params_path = (
        _load_accepted_teacher_run(args.teacher_run)
    )
    teacher_config = _config_from_teacher_run(run_config)
    (
        student_run,
        student_path,
        artifact,
        observations,
        labels,
        student_evaluation,
    ) = _load_bc_run(args.student_run)
    config = _config_from_student_artifact(artifact, teacher_config)
    _validate_bc_teacher_contract(
        artifact, teacher_run, teacher_evaluation, config, teacher_config
    )
    if artifact.metadata.get("stage") != "T5_PHASE_BC":
        raise SystemExit("This audit requires the original phase-conditioned T5 BC policy")
    reference_spec = _reference_spec_from_teacher_run(run_config)
    xml_path = _resolve_xml_path(run_config, args.xml_path)
    output_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else student_run / "failure_audit.json"
    )
    trace_path = (
        args.trace_out.expanduser().resolve()
        if args.trace_out is not None
        else student_run / "failure_audit_trace.npz"
    )

    print(
        f"stage=failure_audit source={student_path} samples={len(observations):,} "
        f"obs={config.student_policy_observation_size} envs={args.envs} steps={args.steps}",
        flush=True,
    )
    offline_report = _offline_error_audit(
        artifact,
        observations,
        labels,
        config.student_action_scale,
        args.phase_bins,
    )
    print(
        "stage=failure_audit_offline "
        f"action_rmse={offline_report['action_error']['rmse']:.7f} "
        f"action_bias={offline_report['action_error']['mean']:+.7f} "
        f"target_rmse_rad={offline_report['target_error_rad']['rmse']:.7f} "
        f"largest_rmse_joint={offline_report['largest_rmse_joint']} "
        f"largest_bias_joint={offline_report['largest_bias_joint']}",
        flush=True,
    )

    configure_cloud_runtime(mujoco_gl=args.mujoco_gl, verbose=True)
    try:
        import jax
        import jax.numpy as jp
        from brax.io import model as model_io
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
    except ImportError as exc:
        raise SystemExit(f"Failure audit requires the mjx312 stack: {exc}") from exc

    reference = build_ik_reference(xml_path, reference_spec)
    nominal_config = replace(config, disturbance_enabled=False)
    env = make_forward_teacher_student_env(
        "dagger",
        config=nominal_config,
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
    obs_mean = jp.asarray(artifact.obs_mean)
    obs_std = jp.asarray(artifact.obs_std)

    noise_reports = []
    for mode in ("bias", "gaussian"):
        for level in args.noise_levels:
            if level == 0.0 and mode == "gaussian":
                continue
            report = _evaluate_noisy_oracle(
                jax,
                jp,
                env,
                teacher_policy,
                args.seed + 10_000,
                args.envs,
                args.steps,
                float(level),
                mode,
            )
            report["mode"] = "baseline" if level == 0.0 else mode
            noise_reports.append(report)
            print(
                f"stage=failure_audit_margin mode={report['mode']} "
                f"action_rms={level:.4f} vx={report['mean_velocity_x']:.4f} "
                f"failure={report['failure_rate']:.3f} "
                f"roll_pitch={report['mean_roll_pitch_rate_rms']:.4f}",
                flush=True,
            )

    paired_values = _paired_divergence_rollout(
        jax,
        jp,
        env,
        teacher_policy,
        student_params,
        obs_mean,
        obs_std,
        args.seed + 20_000,
        args.envs,
        args.steps,
    )
    divergence_report = _divergence_audit(
        paired_values,
        env,
        artifact.obs_mean,
        artifact.obs_std,
        args.q_divergence_rmse,
        args.torso_divergence_m,
        args.nearest_neighbor_samples,
        args.phase_bins,
        args.seed + 30_000,
    )
    np.savez_compressed(
        trace_path,
        observations=paired_values[0],
        phase=paired_values[1],
        student_actions=paired_values[2],
        teacher_corrections=paired_values[3],
        oracle_labels=paired_values[4],
        joint_position_error=paired_values[5],
        joint_velocity_error=paired_values[6],
        torso_position_error=paired_values[7],
        oracle_velocity_x=paired_values[8],
        student_velocity_x=paired_values[9],
        student_roll_pitch_rate_rms=paired_values[10],
        student_failed=paired_values[11],
        alive=paired_values[12],
    )
    q_first = divergence_report["first_q_divergence"]
    nn_report = divergence_report["nearest_neighbor_identifiability"]
    print(
        "stage=failure_audit_divergence "
        f"q_crossing_coverage={q_first['coverage']:.2f} "
        f"median_q_crossing_step={q_first['median_step']} "
        f"oracle_vx={divergence_report['oracle_mean_velocity_x']:.4f} "
        f"student_vx={divergence_report['student_mean_velocity_x']:.4f} "
        f"correction_rmse={divergence_report['student_teacher_action_error']['rmse']:.5f} "
        f"label_drift_rmse={divergence_report['teacher_label_drift_between_branches']['rmse']:.5f}",
        flush=True,
    )
    print(
        "stage=failure_audit_identifiability "
        f"nearest_obs_rms={nn_report.get('nearest_observation_rms', {}).get('mean', 0.0):.5f} "
        f"teacher_disagreement={nn_report.get('teacher_label_disagreement_rms', {}).get('mean', 0.0):.5f} "
        f"student_disagreement={nn_report.get('student_action_disagreement_rms', {}).get('mean', 0.0):.5f} "
        f"excess_fraction={nn_report.get('fraction_teacher_disagreement_gt_student_by_0_01', 0.0):.3f}",
        flush=True,
    )

    final_report = {
        "stage": "T7_FAILURE_AUDIT",
        "teacher_run": str(teacher_run),
        "student_run": str(student_run),
        "student_policy": str(student_path),
        "student_evaluation": student_evaluation,
        "config": asdict(config),
        "offline_error": offline_report,
        "oracle_action_margin": noise_reports,
        "paired_divergence": divergence_report,
        "trace": str(trace_path),
        "interpretation_contract": {
            "small_noise_failure": (
                "The oracle target has insufficient approximation margin; improve the "
                "reference/Teacher robustness before compressing it."
            ),
            "localized_offline_bias": (
                "A small set of joints or phase intervals dominates the error; revise the "
                "action representation or supervised objective."
            ),
            "early_label_drift": (
                "The Teacher becomes a high-gain correction policy immediately after the "
                "Student leaves the oracle trajectory."
            ),
            "nearest_neighbor_label_excess": (
                "Similar deployable observations receive materially different Teacher labels; "
                "the action-distillation target is partially unidentifiable."
            ),
        },
    }
    output_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(
        f"stage=failure_audit_complete report={output_path} trace={trace_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
