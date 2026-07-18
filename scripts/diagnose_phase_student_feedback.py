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
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--student-run", type=Path, required=True)
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=180_000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--mujoco-gl", default="disable")
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


def main(argv=None):
    args = parse_args(argv)
    if args.envs < 2 or args.steps < 3:
        raise SystemExit("--envs must be at least 2 and --steps at least 3")
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
    output_path = (
        args.out.expanduser().resolve()
        if args.out is not None
        else student_run / "feedback_diagnosis.json"
    )
    trace_path = (
        args.trace_out.expanduser().resolve()
        if args.trace_out is not None
        else student_run / "feedback_diagnosis_trace.npz"
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
        config=replace(config, disturbance_enabled=False),
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
        "stage": "T7_FEEDBACK_DIAGNOSIS",
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
    print(
        f"stage=feedback_diagnosis_complete report={output_path} trace={trace_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
