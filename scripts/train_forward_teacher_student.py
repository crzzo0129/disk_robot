from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from disk_robot.gait_speed import plan_forward_gait
from disk_robot.ik_reference import IKReferenceSpec, build_ik_reference
from disk_robot.teacher_student_config import ForwardTeacherStudentConfig
from disk_robot_mjx.pipeline import configure_cloud_runtime, make_network_factory
from disk_robot_mjx.teacher_student_env import DEFAULT_XML, make_forward_teacher_student_env


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train an IK-residual privileged teacher and distill a gait-free forward student."
    )
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--out", type=Path, default=Path("mjx_runs") / "forward_teacher_student")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--teacher-steps", type=int, default=5_000_000)
    parser.add_argument("--teacher-envs", type=int, default=2048)
    parser.add_argument("--teacher-eval-envs", type=int, default=256)
    parser.add_argument("--teacher-evals", type=int, default=6)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--teacher-hidden", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--teacher-learning-rate", type=float, default=1e-4)
    parser.add_argument("--teacher-entropy-cost", type=float, default=1e-3)
    parser.add_argument("--teacher-unroll-length", type=int, default=20)
    parser.add_argument("--teacher-batch-size", type=int, default=256)
    parser.add_argument("--teacher-minibatches", type=int, default=32)
    parser.add_argument("--teacher-updates-per-batch", type=int, default=4)
    parser.add_argument(
        "--teacher-zero-policy-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the PPO actor output layer to zero residual commands.",
    )
    parser.add_argument("--residual-filter-alpha", type=float, default=0.15)
    parser.add_argument("--penalty-residual", type=float, default=0.20)
    parser.add_argument("--penalty-residual-rate", type=float, default=0.05)
    parser.add_argument(
        "--residual-scale-multiplier",
        type=float,
        default=1.0,
        help="Multiply the per-joint teacher residual limits without changing IK or student actions.",
    )
    parser.add_argument("--teacher-restore", type=Path, default=None)
    parser.add_argument(
        "--teacher-selection-mode",
        choices=("improve", "preserve", "robust"),
        default="improve",
        help="Select by nominal improvement, T1b preservation, or the T2 nominal/disturbed gate.",
    )
    parser.add_argument("--teacher-disturbances", action="store_true")
    parser.add_argument("--push-step-min", type=int, default=100)
    parser.add_argument("--push-step-max", type=int, default=350)
    parser.add_argument("--push-velocity-x", type=float, default=0.50)
    parser.add_argument("--push-velocity-y", type=float, default=0.40)
    parser.add_argument("--motor-strength-min", type=float, default=0.85)
    parser.add_argument("--motor-strength-max", type=float, default=1.15)
    parser.add_argument("--control-delay-probability", type=float, default=0.50)
    parser.add_argument("--disturbance-reset-joint-noise", type=float, default=0.030)
    parser.add_argument("--disturbance-reset-height-noise", type=float, default=0.005)
    parser.add_argument("--recovery-window-steps", type=int, default=100)
    parser.add_argument("--recovery-velocity-ema-alpha", type=float, default=0.10)
    parser.add_argument("--recovery-forward-tolerance", type=float, default=0.04)
    parser.add_argument("--recovery-lateral-tolerance", type=float, default=0.04)
    parser.add_argument("--recovery-required-steps", type=int, default=4)
    parser.add_argument("--min-disturbed-score-improvement", type=float, default=0.02)
    parser.add_argument("--min-accepted-teacher-vx", type=float, default=None)
    parser.add_argument("--max-accepted-teacher-failure-rate", type=float, default=0.10)
    parser.add_argument("--max-accepted-teacher-velocity-error", type=float, default=0.03)
    parser.add_argument("--max-accepted-teacher-roll-pitch-rate", type=float, default=0.50)
    parser.add_argument("--max-accepted-teacher-lateral-speed", type=float, default=0.03)
    parser.add_argument("--max-accepted-teacher-yaw-rate", type=float, default=0.25)
    parser.add_argument(
        "--allow-ik-baseline-teacher",
        action="store_true",
        help="Allow distillation from zero-residual IK when PPO is not the selected teacher.",
    )

    parser.add_argument("--rollout-envs", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=500)
    parser.add_argument("--dataset-samples", type=int, default=131_072)
    parser.add_argument("--student-hidden", type=int, nargs="+", default=[256, 128, 128])
    parser.add_argument("--student-updates", type=int, default=20_000)
    parser.add_argument("--student-batch-size", type=int, default=1024)
    parser.add_argument("--student-learning-rate", type=float, default=3e-4)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-samples", type=int, default=65_536)
    parser.add_argument("--dagger-updates", type=int, default=5_000)
    parser.add_argument("--save-dataset", action="store_true")

    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--min-accepted-vx", type=float, default=None)
    parser.add_argument("--max-accepted-failure-rate", type=float, default=0.10)
    parser.add_argument("--max-accepted-velocity-error", type=float, default=0.03)
    parser.add_argument("--max-accepted-roll-pitch-rate", type=float, default=0.60)
    parser.add_argument("--max-accepted-lateral-speed", type=float, default=0.03)
    parser.add_argument("--max-accepted-yaw-rate", type=float, default=0.25)
    parser.add_argument("--strict-acceptance", action="store_true")
    parser.add_argument(
        "--teacher-only",
        action="store_true",
        help="Stop after teacher selection and acceptance without BC or DAgger.",
    )

    parser.add_argument("--ik-samples", type=int, default=256)
    parser.add_argument(
        "--ik-speed-mode",
        choices=("command", "manual"),
        default="command",
        help="Derive the IK gait from --command-vx or use explicit IK parameters.",
    )
    parser.add_argument("--ik-frequency", type=float, default=0.8)
    parser.add_argument("--ik-stride", type=float, default=0.04)
    parser.add_argument("--ik-height", type=float, default=0.025)
    parser.add_argument("--ik-duty", type=float, default=0.72)
    parser.add_argument("--command-vx", type=float, default=0.08)
    parser.add_argument("--kp", type=float, default=10.0)
    parser.add_argument("--kd", type=float, default=0.4)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    parser.add_argument("--startup-steps", type=int, default=25)

    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--no-xla-triton", dest="xla_triton", action="store_false")
    parser.set_defaults(xla_triton=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _resolve_reference_spec(args) -> tuple[IKReferenceSpec, dict]:
    if args.ik_speed_mode == "command":
        try:
            plan = plan_forward_gait(args.command_vx)
        except ValueError as exc:
            raise SystemExit(f"cannot build command-conditioned IK reference: {exc}") from exc
        frequency = plan.frequency
        stride = plan.stride_length
        height = args.ik_height * plan.motion_scale
        source = {
            "mode": "command",
            "target_speed": plan.target_speed,
            "calibration": "candidate_structure_v1",
        }
    else:
        frequency = args.ik_frequency
        stride = args.ik_stride
        height = args.ik_height
        source = {"mode": "manual"}

    spec = IKReferenceSpec(
        samples=args.ik_samples,
        frequency=frequency,
        stride_length=stride,
        step_height=height,
        duty=args.ik_duty,
        mode="trot",
    )
    return spec, source


def _resolve_acceptance_thresholds(args) -> None:
    minimum_velocity = max(0.02, args.command_vx - 0.02)
    if args.min_accepted_teacher_vx is None:
        args.min_accepted_teacher_vx = minimum_velocity
    if args.min_accepted_vx is None:
        args.min_accepted_vx = minimum_velocity


def _resolve_latest_checkpoint(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        return path
    numbered = sorted(
        (child for child in path.iterdir() if child.is_dir() and child.name.isdigit()),
        key=lambda child: int(child.name),
    )
    return numbered[-1] if numbered else path


def _metric_scalar(value):
    try:
        import jax

        value = jax.device_get(value)
    except ImportError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _teacher_eval_summary(metrics, command_vx):
    episode_length = _metric_scalar(metrics.get("eval/avg_episode_length"))
    if episode_length is None or episode_length <= 0.0:
        return {}
    summary = {"episode_length": episode_length}
    for output_name, metric_name in (
        ("reward_per_step", "eval/episode_reward"),
        ("mean_velocity_x", "eval/episode_velocity_x"),
        ("mean_instantaneous_velocity_error", "eval/episode_velocity_error"),
        ("mean_abs_velocity_y", "eval/episode_abs_velocity_y"),
        ("mean_abs_yaw_rate", "eval/episode_abs_yaw_rate"),
        ("mean_roll_pitch_rate_rms", "eval/episode_roll_pitch_rate_rms"),
    ):
        value = _metric_scalar(metrics.get(metric_name))
        if value is not None:
            summary[output_name] = value / episode_length
    failed = _metric_scalar(metrics.get("eval/episode_failed"))
    if failed is not None:
        summary["failure_rate"] = failed
    if "mean_velocity_x" in summary:
        summary["mean_velocity_error"] = abs(summary["mean_velocity_x"] - command_vx)
    return summary


def _teacher_progress(step, metrics, command_vx):
    summary = _teacher_eval_summary(metrics, command_vx)
    print(
        f"stage=teacher_eval step={int(step):,} "
        f"reward={summary.get('reward_per_step', float('nan')):.4f} "
        f"vx={summary.get('mean_velocity_x', float('nan')):.4f} "
        f"vx_error={summary.get('mean_velocity_error', float('nan')):.4f} "
        f"failure={summary.get('failure_rate', float('nan')):.3f} "
        f"roll_pitch={summary.get('mean_roll_pitch_rate_rms', float('nan')):.4f} "
        f"abs_vy={summary.get('mean_abs_velocity_y', float('nan')):.4f} "
        f"abs_yaw={summary.get('mean_abs_yaw_rate', float('nan')):.4f}",
        flush=True,
    )


def _print_evaluation_summary(stage, report, mode=None):
    prefix = f"stage={stage}"
    if mode is not None:
        prefix += f" mode={mode}"
    values = [
        f"reward={report.get('reward_per_step', float('nan')):.4f}",
        f"vx={report.get('mean_velocity_x', float('nan')):.4f}",
        f"distance={report.get('mean_forward_distance', float('nan')):.4f}",
        f"failure={report.get('failure_rate', float('nan')):.3f}",
        f"roll_pitch={report.get('mean_roll_pitch_rate_rms', float('nan')):.4f}",
        f"abs_vy={report.get('mean_abs_velocity_y', float('nan')):.4f}",
        f"abs_yaw={report.get('mean_abs_yaw_rate', float('nan')):.4f}",
        f"disk={report.get('mean_disk_contacts', float('nan')):.4f}",
    ]
    if mode == "disturbed":
        values.extend(
            (
                f"post_error={report.get('mean_post_push_velocity_error', float('nan')):.4f}",
                f"recovery_s={report.get('mean_recovery_time', float('nan')):.3f}",
                f"push_coverage={report.get('push_coverage', float('nan')):.2f}",
            )
        )
    print(f"{prefix} {' '.join(values)}", flush=True)


def _print_teacher_comparison(mode, ppo_report, baseline_report, score_gain=None):
    values = []
    if score_gain is not None:
        values.append(f"score_gain={score_gain:+.4f}")
    values.extend(
        (
            f"delta_vx={ppo_report['mean_velocity_x'] - baseline_report['mean_velocity_x']:+.4f}",
            f"delta_failure={ppo_report['failure_rate'] - baseline_report['failure_rate']:+.3f}",
            f"delta_roll_pitch={ppo_report['mean_roll_pitch_rate_rms'] - baseline_report['mean_roll_pitch_rate_rms']:+.4f}",
        )
    )
    if mode == "disturbed":
        values.extend(
            (
                f"delta_post_error={ppo_report['mean_post_push_velocity_error'] - baseline_report['mean_post_push_velocity_error']:+.4f}",
                f"delta_recovery_s={ppo_report['mean_recovery_time'] - baseline_report['mean_recovery_time']:+.3f}",
                f"delta_distance={ppo_report['mean_forward_distance'] - baseline_report['mean_forward_distance']:+.4f}",
                f"delta_disk={ppo_report['mean_disk_contacts'] - baseline_report['mean_disk_contacts']:+.4f}",
            )
        )
    print(f"stage=teacher_comparison mode={mode} {' '.join(values)}", flush=True)


def _teacher_selection_score(report):
    return (
        report.get("reward_per_step", 0.0)
        - 2.0 * report.get("failure_rate", 0.0)
        - 0.5 * report.get("mean_velocity_error", 0.0)
        - 0.1 * report.get("mean_roll_pitch_rate_rms", 0.0)
        - 0.2 * report.get("mean_abs_velocity_y", 0.0)
        - 0.1 * report.get("mean_abs_yaw_rate", 0.0)
    )


def _ppo_preserves_baseline(ppo_report, baseline_report):
    return (
        ppo_report["mean_velocity_x"] >= baseline_report["mean_velocity_x"] - 0.01
        and ppo_report["failure_rate"] <= baseline_report["failure_rate"] + 0.02
        and ppo_report["mean_roll_pitch_rate_rms"]
        <= baseline_report["mean_roll_pitch_rate_rms"] + 0.10
        and ppo_report["mean_abs_velocity_y"]
        <= baseline_report["mean_abs_velocity_y"] + 0.01
        and ppo_report["mean_abs_yaw_rate"]
        <= baseline_report["mean_abs_yaw_rate"] + 0.05
    )


def _should_select_ppo(selection_mode, ppo_score, baseline_score, preserves_baseline):
    return preserves_baseline and (
        selection_mode == "preserve" or ppo_score > baseline_score
    )


def _disturbed_teacher_score(report):
    return (
        _teacher_selection_score(report)
        - 3.0 * report.get("failure_rate", 0.0)
        - 1.0 * report.get("mean_post_push_velocity_error", 0.0)
        - 0.5 * report.get("mean_recovery_time", 0.0)
        - 0.5 * report.get("mean_disk_contacts", 0.0)
        + 0.2 * report.get("mean_forward_distance", 0.0)
    )


def _ppo_improves_disturbed_baseline(ppo_report, baseline_report, minimum_score_gain=0.02):
    ppo_score = _disturbed_teacher_score(ppo_report)
    baseline_score = _disturbed_teacher_score(baseline_report)
    preserves_safety = (
        ppo_report["failure_rate"] <= baseline_report["failure_rate"]
        and ppo_report["mean_post_push_velocity_error"]
        <= baseline_report["mean_post_push_velocity_error"]
        and ppo_report["mean_recovery_time"] <= baseline_report["mean_recovery_time"]
        and ppo_report["mean_disk_contacts"] <= baseline_report["mean_disk_contacts"] + 0.01
        and ppo_report["mean_forward_distance"]
        >= baseline_report["mean_forward_distance"] - 0.02
    )
    return preserves_safety and ppo_score >= baseline_score + minimum_score_gain


def _estimate_brax_timesteps(
    requested_steps, num_evals, batch_size, num_minibatches, unroll_length
):
    intervals = max(int(num_evals) - 1, 1)
    step_quantum = int(batch_size) * int(num_minibatches) * int(unroll_length)
    batches_per_interval = max(1, math.ceil(requested_steps / intervals / step_quantum))
    return {
        "requested_steps": int(requested_steps),
        "step_quantum": step_quantum,
        "evaluation_intervals": intervals,
        "estimated_effective_steps": step_quantum * batches_per_interval * intervals,
    }


def _teacher_gate_acceptance(
    selection_mode,
    selected_source,
    absolute_thresholds_passed,
    preserves_nominal,
    improves_disturbed,
    allow_ik_baseline_teacher=False,
):
    if selection_mode == "preserve":
        return selected_source == "ppo" and preserves_nominal
    if selection_mode == "robust":
        return selected_source == "ppo" and preserves_nominal and improves_disturbed
    return absolute_thresholds_passed and (
        selected_source == "ppo" or allow_ik_baseline_teacher
    )


def _student_init(jax, jp, key, layer_sizes):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for layer_key, input_size, output_size in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        scale = np.sqrt(2.0 / (input_size + output_size))
        weight = scale * jax.random.normal(layer_key, (input_size, output_size))
        bias = jp.zeros((output_size,))
        params.append((weight, bias))
    return tuple(params)


def _student_apply(jp, params, normalized_obs):
    value = normalized_obs
    for weight, bias in params[:-1]:
        preactivation = value @ weight + bias
        value = jp.where(preactivation > 0.0, preactivation, jp.expm1(preactivation))
    weight, bias = params[-1]
    return jp.tanh(value @ weight + bias)


def _normalized_student_apply(jp, params, obs, obs_mean, obs_std):
    normalized = jp.clip((obs - obs_mean) / obs_std, -10.0, 10.0)
    return _student_apply(jp, params, normalized)


def _save_student_policy(path: Path, params, obs_mean, obs_std, metadata):
    import jax

    arrays = {
        "obs_mean": np.asarray(jax.device_get(obs_mean)),
        "obs_std": np.asarray(jax.device_get(obs_std)),
    }
    for index, (weight, bias) in enumerate(params):
        arrays[f"weight_{index}"] = np.asarray(jax.device_get(weight))
        arrays[f"bias_{index}"] = np.asarray(jax.device_get(bias))
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _sample_dataset_rows(observations, labels, valid, take, seed):
    observations = observations[valid]
    labels = labels[valid]
    if take >= len(observations):
        return observations, labels
    indices = np.random.default_rng(seed).choice(len(observations), size=take, replace=False)
    return observations[indices], labels[indices]


def _collect_teacher_dataset(
    jax,
    jp,
    env,
    teacher_policy,
    seed,
    env_count,
    horizon,
    requested_samples,
    student_observation_key="student_obs",
):
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step))
    label_batch = jax.jit(jax.vmap(env.teacher_action_to_student_action))
    observations = []
    labels = []
    collected = 0
    batch_index = 0

    while collected < requested_samples:
        reset_key = jax.random.PRNGKey(seed + batch_index)
        state = reset_batch(jax.random.split(reset_key, env_count))

        def rollout_step(carry, _):
            current_state, policy_key = carry
            policy_key, action_key = jax.random.split(policy_key)
            residual_action, _ = teacher_policy(current_state.info["teacher_obs"], action_key)
            student_label = label_batch(current_state, residual_action)
            valid = 1.0 - current_state.done
            next_state = step_batch(current_state, residual_action)
            return (next_state, policy_key), (
                current_state.info[student_observation_key],
                student_label,
                valid,
            )

        (_, _), (obs, target, valid) = jax.lax.scan(
            rollout_step,
            (state, jax.random.PRNGKey(seed + 100_000 + batch_index)),
            (),
            length=horizon,
        )
        obs = np.asarray(jax.device_get(obs)).reshape(-1, obs.shape[-1])
        target = np.asarray(jax.device_get(target)).reshape(-1, target.shape[-1])
        valid = np.asarray(jax.device_get(valid)).reshape(-1) > 0.5
        take = min(int(np.sum(valid)), requested_samples - collected)
        obs, target = _sample_dataset_rows(
            obs,
            target,
            valid,
            take,
            seed + 1_000_000 + batch_index,
        )
        observations.append(obs)
        labels.append(target)
        collected += take
        batch_index += 1
        print(f"stage=teacher_dataset samples={collected:,}/{requested_samples:,}", flush=True)

    return np.concatenate(observations), np.concatenate(labels)


def _evaluate_teacher(jax, env, teacher_policy, seed, env_count, horizon):
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step))
    state = reset_batch(jax.random.split(jax.random.PRNGKey(seed), env_count))

    def eval_step(carry, _):
        current_state, policy_key = carry
        policy_key, action_key = jax.random.split(policy_key)
        action, _ = teacher_policy(current_state.obs, action_key)
        next_state = step_batch(current_state, action)
        alive = 1.0 - current_state.done
        return (next_state, policy_key), (
            next_state.reward,
            next_state.metrics["velocity_x"],
            next_state.metrics["velocity_y"],
            next_state.metrics["abs_velocity_y"],
            next_state.metrics["abs_yaw_rate"],
            next_state.metrics["velocity_error"],
            next_state.metrics["smoothed_velocity_error"],
            next_state.metrics["roll_pitch_rate_rms"],
            next_state.metrics["disk_contact_count"],
            next_state.metrics["push_applied"],
            next_state.metrics["post_push"],
            next_state.metrics["recovery_ready"],
            next_state.metrics["failed"],
            alive,
        )

    (_, _), values = jax.lax.scan(
        eval_step,
        (state, jax.random.PRNGKey(seed + 1)),
        (),
        length=horizon,
    )
    (
        reward,
        vx,
        vy,
        abs_vy,
        abs_yaw_rate,
        velocity_error,
        smoothed_velocity_error,
        roll_pitch_rate,
        disk_contact,
        push_applied,
        post_push,
        recovery_ready,
        failed,
        alive,
    ) = [np.asarray(jax.device_get(value)) for value in values]
    denominator = max(float(np.sum(alive)), 1.0)
    post_push_alive = post_push * alive
    post_push_denominator = max(float(np.sum(post_push_alive)), 1.0)
    has_push = np.max(push_applied, axis=0) > 0.5
    recovery_candidates = (recovery_ready > 0.5) & (post_push > 0.5) & (alive > 0.5)
    first_recovery = np.argmax(recovery_candidates, axis=0)
    recovered = np.any(recovery_candidates, axis=0)
    push_index = np.argmax(push_applied, axis=0)
    recovery_steps = np.where(
        recovered,
        np.maximum(first_recovery - push_index, 0),
        env.config.recovery_window_steps,
    )
    mean_recovery_time = (
        float(np.mean(recovery_steps[has_push])) * env.dt if np.any(has_push) else 0.0
    )
    mean_velocity_x = float(np.sum(vx * alive) / denominator)
    return {
        "reward_per_step": float(np.sum(reward * alive) / denominator),
        "mean_velocity_x": mean_velocity_x,
        "mean_forward_distance": float(np.mean(np.sum(vx * alive, axis=0)) * env.dt),
        "mean_lateral_distance": float(np.mean(np.sum(vy * alive, axis=0)) * env.dt),
        "mean_abs_velocity_y": float(np.sum(abs_vy * alive) / denominator),
        "mean_abs_yaw_rate": float(np.sum(abs_yaw_rate * alive) / denominator),
        "mean_velocity_error": abs(mean_velocity_x - env.config.command_vx),
        "mean_instantaneous_velocity_error": float(
            np.sum(velocity_error * alive) / denominator
        ),
        "mean_roll_pitch_rate_rms": float(np.sum(roll_pitch_rate * alive) / denominator),
        "mean_disk_contacts": float(np.sum(disk_contact * alive) / denominator),
        "mean_post_push_velocity_error": float(
            np.sum(smoothed_velocity_error * post_push_alive) / post_push_denominator
        ),
        "mean_post_push_instantaneous_velocity_error": float(
            np.sum(velocity_error * post_push_alive) / post_push_denominator
        ),
        "mean_post_push_abs_velocity_y": float(
            np.sum(abs_vy * post_push_alive) / post_push_denominator
        ),
        "mean_recovery_time": mean_recovery_time,
        "push_coverage": float(np.mean(has_push)),
        "failure_rate": float(np.mean(np.max(failed, axis=0))),
        "mean_alive_steps": float(np.mean(np.sum(alive, axis=0))),
    }


def _evaluate_oracle_student(jax, env, teacher_policy, seed, env_count, horizon):
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step))
    label_batch = jax.jit(jax.vmap(env.teacher_action_to_student_action))
    state = reset_batch(jax.random.split(jax.random.PRNGKey(seed), env_count))

    def eval_step(carry, _):
        current_state, policy_key = carry
        policy_key, action_key = jax.random.split(policy_key)
        residual_action, _ = teacher_policy(
            current_state.info["teacher_obs"], action_key
        )
        direct_action = label_batch(current_state, residual_action)
        next_state = step_batch(current_state, direct_action)
        alive = 1.0 - current_state.done
        return (next_state, policy_key), (
            next_state.reward,
            next_state.metrics["velocity_x"],
            next_state.metrics["velocity_y"],
            next_state.metrics["abs_velocity_y"],
            next_state.metrics["abs_yaw_rate"],
            next_state.metrics["velocity_error"],
            next_state.metrics["smoothed_velocity_error"],
            next_state.metrics["roll_pitch_rate_rms"],
            next_state.metrics["disk_contact_count"],
            next_state.metrics["push_applied"],
            next_state.metrics["post_push"],
            next_state.metrics["recovery_ready"],
            next_state.metrics["failed"],
            alive,
        )

    (_, _), values = jax.lax.scan(
        eval_step,
        (state, jax.random.PRNGKey(seed + 1)),
        (),
        length=horizon,
    )
    (
        reward,
        vx,
        vy,
        abs_vy,
        abs_yaw_rate,
        velocity_error,
        smoothed_velocity_error,
        roll_pitch_rate,
        disk_contact,
        push_applied,
        post_push,
        recovery_ready,
        failed,
        alive,
    ) = [np.asarray(jax.device_get(value)) for value in values]
    denominator = max(float(np.sum(alive)), 1.0)
    post_push_alive = post_push * alive
    post_push_denominator = max(float(np.sum(post_push_alive)), 1.0)
    has_push = np.max(push_applied, axis=0) > 0.5
    recovery_candidates = (recovery_ready > 0.5) & (post_push > 0.5) & (alive > 0.5)
    first_recovery = np.argmax(recovery_candidates, axis=0)
    recovered = np.any(recovery_candidates, axis=0)
    push_index = np.argmax(push_applied, axis=0)
    recovery_steps = np.where(
        recovered,
        np.maximum(first_recovery - push_index, 0),
        env.config.recovery_window_steps,
    )
    mean_recovery_time = (
        float(np.mean(recovery_steps[has_push])) * env.dt if np.any(has_push) else 0.0
    )
    mean_velocity_x = float(np.sum(vx * alive) / denominator)
    return {
        "reward_per_step": float(np.sum(reward * alive) / denominator),
        "mean_velocity_x": mean_velocity_x,
        "mean_forward_distance": float(np.mean(np.sum(vx * alive, axis=0)) * env.dt),
        "mean_lateral_distance": float(np.mean(np.sum(vy * alive, axis=0)) * env.dt),
        "mean_abs_velocity_y": float(np.sum(abs_vy * alive) / denominator),
        "mean_abs_yaw_rate": float(np.sum(abs_yaw_rate * alive) / denominator),
        "mean_velocity_error": abs(mean_velocity_x - env.config.command_vx),
        "mean_instantaneous_velocity_error": float(
            np.sum(velocity_error * alive) / denominator
        ),
        "mean_roll_pitch_rate_rms": float(np.sum(roll_pitch_rate * alive) / denominator),
        "mean_disk_contacts": float(np.sum(disk_contact * alive) / denominator),
        "mean_post_push_velocity_error": float(
            np.sum(smoothed_velocity_error * post_push_alive) / post_push_denominator
        ),
        "mean_post_push_instantaneous_velocity_error": float(
            np.sum(velocity_error * post_push_alive) / post_push_denominator
        ),
        "mean_post_push_abs_velocity_y": float(
            np.sum(abs_vy * post_push_alive) / post_push_denominator
        ),
        "mean_recovery_time": mean_recovery_time,
        "push_coverage": float(np.mean(has_push)),
        "failure_rate": float(np.mean(np.max(failed, axis=0))),
        "mean_alive_steps": float(np.mean(np.sum(alive, axis=0))),
    }


def _train_student(
    jax,
    jp,
    optax,
    params,
    observations,
    labels,
    obs_mean,
    obs_std,
    updates,
    batch_size,
    learning_rate,
    seed,
    stage,
    anchor_params=None,
    anchor_weight=0.0,
):
    obs = jp.asarray(observations)
    target = jp.asarray(labels)
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)

    def loss_fn(current_params, batch_obs, batch_target):
        prediction = _normalized_student_apply(jp, current_params, batch_obs, obs_mean, obs_std)
        teacher_loss = jp.mean(jp.square(prediction - batch_target))
        if anchor_params is None or anchor_weight <= 0.0:
            return teacher_loss
        anchor_prediction = jax.lax.stop_gradient(
            _normalized_student_apply(jp, anchor_params, batch_obs, obs_mean, obs_std)
        )
        anchor_loss = jp.mean(jp.square(prediction - anchor_prediction))
        return teacher_loss + anchor_weight * anchor_loss

    @jax.jit
    def update(current_params, current_optimizer_state, batch_obs, batch_target):
        loss, gradients = jax.value_and_grad(loss_fn)(current_params, batch_obs, batch_target)
        updates_value, next_optimizer_state = optimizer.update(
            gradients, current_optimizer_state, current_params
        )
        next_params = optax.apply_updates(current_params, updates_value)
        return next_params, next_optimizer_state, loss

    rng = jax.random.PRNGKey(seed)
    report_every = max(1, updates // 20)
    for update_index in range(updates):
        rng, index_key = jax.random.split(rng)
        indices = jax.random.randint(index_key, (batch_size,), 0, obs.shape[0])
        params, optimizer_state, loss = update(params, optimizer_state, obs[indices], target[indices])
        if update_index % report_every == 0 or update_index + 1 == updates:
            print(
                f"stage={stage} update={update_index + 1:,}/{updates:,} loss={float(jax.device_get(loss)):.7f}",
                flush=True,
            )
    return params


def _collect_dagger_dataset(
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
    requested_samples,
    teacher_rollout_blend=0.0,
):
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step))
    label_batch = jax.jit(jax.vmap(env.teacher_action_to_student_action))
    student_policy = jax.jit(
        lambda obs: _normalized_student_apply(jp, student_params, obs, obs_mean, obs_std)
    )
    observations = []
    labels = []
    collected = 0
    batch_index = 0

    while collected < requested_samples:
        state = reset_batch(jax.random.split(jax.random.PRNGKey(seed + batch_index), env_count))

        def rollout_step(carry, _):
            current_state, policy_key = carry
            policy_key, teacher_key = jax.random.split(policy_key)
            student_action = student_policy(current_state.obs)
            teacher_residual, _ = teacher_policy(current_state.info["teacher_obs"], teacher_key)
            teacher_label = label_batch(current_state, teacher_residual)
            rollout_action = (
                (1.0 - teacher_rollout_blend) * student_action
                + teacher_rollout_blend * teacher_label
            )
            valid = 1.0 - current_state.done
            next_state = step_batch(current_state, rollout_action)
            return (next_state, policy_key), (current_state.obs, teacher_label, valid)

        (_, _), (obs, target, valid) = jax.lax.scan(
            rollout_step,
            (state, jax.random.PRNGKey(seed + 100_000 + batch_index)),
            (),
            length=horizon,
        )
        obs = np.asarray(jax.device_get(obs)).reshape(-1, obs.shape[-1])
        target = np.asarray(jax.device_get(target)).reshape(-1, target.shape[-1])
        valid = np.asarray(jax.device_get(valid)).reshape(-1) > 0.5
        take = min(int(np.sum(valid)), requested_samples - collected)
        obs, target = _sample_dataset_rows(
            obs,
            target,
            valid,
            take,
            seed + 1_000_000 + batch_index,
        )
        observations.append(obs)
        labels.append(target)
        collected += take
        batch_index += 1
        print(f"stage=dagger_dataset samples={collected:,}/{requested_samples:,}", flush=True)

    return np.concatenate(observations), np.concatenate(labels)


def _evaluate_student(jax, jp, env, params, obs_mean, obs_std, seed, env_count, horizon):
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step))
    policy = jax.jit(lambda obs: _normalized_student_apply(jp, params, obs, obs_mean, obs_std))
    state = reset_batch(jax.random.split(jax.random.PRNGKey(seed), env_count))

    def eval_step(current_state, _):
        action = policy(current_state.obs)
        next_state = step_batch(current_state, action)
        alive = 1.0 - current_state.done
        return next_state, (
            next_state.reward,
            next_state.metrics["velocity_x"],
            next_state.metrics["velocity_y"],
            next_state.metrics["abs_velocity_y"],
            next_state.metrics["abs_yaw_rate"],
            next_state.metrics["velocity_error"],
            next_state.metrics["smoothed_velocity_error"],
            next_state.metrics["roll_pitch_rate_rms"],
            next_state.metrics["disk_contact_count"],
            next_state.metrics["push_applied"],
            next_state.metrics["post_push"],
            next_state.metrics["recovery_ready"],
            next_state.metrics["failed"],
            alive,
        )

    _, values = jax.lax.scan(eval_step, state, (), length=horizon)
    (
        reward,
        vx,
        vy,
        abs_vy,
        abs_yaw_rate,
        velocity_error,
        smoothed_velocity_error,
        roll_pitch_rate,
        disk_contact,
        push_applied,
        post_push,
        recovery_ready,
        failed,
        alive,
    ) = [np.asarray(jax.device_get(value)) for value in values]
    denominator = max(float(np.sum(alive)), 1.0)
    post_push_alive = post_push * alive
    post_push_denominator = max(float(np.sum(post_push_alive)), 1.0)
    has_push = np.max(push_applied, axis=0) > 0.5
    recovery_candidates = (recovery_ready > 0.5) & (post_push > 0.5) & (alive > 0.5)
    first_recovery = np.argmax(recovery_candidates, axis=0)
    recovered = np.any(recovery_candidates, axis=0)
    push_index = np.argmax(push_applied, axis=0)
    recovery_steps = np.where(
        recovered,
        np.maximum(first_recovery - push_index, 0),
        env.config.recovery_window_steps,
    )
    mean_recovery_time = (
        float(np.mean(recovery_steps[has_push])) * env.dt if np.any(has_push) else 0.0
    )
    failed_any = np.max(failed, axis=0)
    mean_velocity_x = float(np.sum(vx * alive) / denominator)
    return {
        "reward_per_step": float(np.sum(reward * alive) / denominator),
        "mean_velocity_x": mean_velocity_x,
        "mean_forward_distance": float(np.mean(np.sum(vx * alive, axis=0)) * env.dt),
        "mean_lateral_distance": float(np.mean(np.sum(vy * alive, axis=0)) * env.dt),
        "mean_abs_velocity_y": float(np.sum(abs_vy * alive) / denominator),
        "mean_abs_yaw_rate": float(np.sum(abs_yaw_rate * alive) / denominator),
        "mean_velocity_error": abs(mean_velocity_x - env.config.command_vx),
        "mean_instantaneous_velocity_error": float(
            np.sum(velocity_error * alive) / denominator
        ),
        "mean_roll_pitch_rate_rms": float(np.sum(roll_pitch_rate * alive) / denominator),
        "mean_disk_contacts": float(np.sum(disk_contact * alive) / denominator),
        "mean_post_push_velocity_error": float(
            np.sum(smoothed_velocity_error * post_push_alive) / post_push_denominator
        ),
        "mean_post_push_instantaneous_velocity_error": float(
            np.sum(velocity_error * post_push_alive) / post_push_denominator
        ),
        "mean_post_push_abs_velocity_y": float(
            np.sum(abs_vy * post_push_alive) / post_push_denominator
        ),
        "mean_recovery_time": mean_recovery_time,
        "push_coverage": float(np.mean(has_push)),
        "failure_rate": float(np.mean(failed_any)),
        "mean_alive_steps": float(np.mean(np.sum(alive, axis=0))),
    }


def main(argv=None):
    args = parse_args(argv)
    _resolve_acceptance_thresholds(args)
    if not 0.0 < args.residual_scale_multiplier <= 1.0:
        raise SystemExit("--residual-scale-multiplier must be in (0, 1]")
    if not 0.0 < args.residual_filter_alpha <= 1.0:
        raise SystemExit("--residual-filter-alpha must be in (0, 1]")
    if args.penalty_residual < 0.0 or args.penalty_residual_rate < 0.0:
        raise SystemExit("residual penalties must be non-negative")
    if args.teacher_selection_mode == "preserve" and not args.teacher_only:
        raise SystemExit("--teacher-selection-mode preserve requires --teacher-only")
    if args.teacher_selection_mode == "robust" and not args.teacher_disturbances:
        raise SystemExit("--teacher-selection-mode robust requires --teacher-disturbances")
    if not 0.0 <= args.control_delay_probability <= 1.0:
        raise SystemExit("--control-delay-probability must be in [0, 1]")
    if not 0 < args.push_step_min <= args.push_step_max < args.episode_length:
        raise SystemExit("push steps must satisfy 0 < min <= max < episode length")
    if not 0.0 < args.motor_strength_min <= args.motor_strength_max:
        raise SystemExit("motor strength range must be positive and ordered")
    if not 0.0 < args.recovery_velocity_ema_alpha <= 1.0:
        raise SystemExit("--recovery-velocity-ema-alpha must be in (0, 1]")
    if args.recovery_forward_tolerance <= 0.0 or args.recovery_lateral_tolerance <= 0.0:
        raise SystemExit("recovery velocity tolerances must be positive")
    if args.recovery_required_steps < 1:
        raise SystemExit("--recovery-required-steps must be at least 1")
    if args.smoke:
        args.teacher_steps = min(args.teacher_steps, 20_000)
        args.teacher_envs = min(args.teacher_envs, 64)
        args.teacher_eval_envs = min(args.teacher_eval_envs, 32)
        args.teacher_evals = 2
        args.teacher_unroll_length = min(args.teacher_unroll_length, 10)
        args.teacher_batch_size = min(args.teacher_batch_size, 32)
        args.teacher_minibatches = min(args.teacher_minibatches, 4)
        args.teacher_updates_per_batch = min(args.teacher_updates_per_batch, 2)
        args.dataset_samples = min(args.dataset_samples, 2_048)
        args.rollout_envs = min(args.rollout_envs, 16)
        args.rollout_horizon = min(args.rollout_horizon, 64)
        args.student_updates = min(args.student_updates, 20)
        args.dagger_rounds = min(args.dagger_rounds, 1)
        args.dagger_samples = min(args.dagger_samples, 1_024)
        args.dagger_updates = min(args.dagger_updates, 10)
        args.eval_envs = min(args.eval_envs, 16)

    training_plan = _estimate_brax_timesteps(
        args.teacher_steps,
        args.teacher_evals,
        args.teacher_batch_size,
        args.teacher_minibatches,
        args.teacher_unroll_length,
    )
    print(
        "stage=teacher_step_plan "
        f"requested={training_plan['requested_steps']:,} "
        f"effective={training_plan['estimated_effective_steps']:,} "
        f"quantum={training_plan['step_quantum']:,} "
        f"eval_intervals={training_plan['evaluation_intervals']}",
        flush=True,
    )

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
        from brax.training.agents.ppo import train as ppo
    except ImportError as exc:
        raise SystemExit(
            "Activate the offline mjx312 environment with jax, mujoco, brax, and optax installed."
        ) from exc

    args.xml_path = args.xml_path.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    teacher_dir = args.out / "teacher"
    teacher_dir.mkdir(exist_ok=True)

    config = ForwardTeacherStudentConfig(
        max_episode_steps=args.episode_length,
        command_vx=args.command_vx,
        actuator_kp=args.kp,
        actuator_kd=args.kd,
        torque_limit=args.torque_limit,
        startup_blend_steps=args.startup_steps,
        residual_filter_alpha=args.residual_filter_alpha,
        penalty_residual=args.penalty_residual,
        penalty_residual_rate=args.penalty_residual_rate,
        disturbance_enabled=args.teacher_disturbances,
        push_step_min=args.push_step_min,
        push_step_max=args.push_step_max,
        push_velocity_x=args.push_velocity_x,
        push_velocity_y=args.push_velocity_y,
        motor_strength_min=args.motor_strength_min,
        motor_strength_max=args.motor_strength_max,
        control_delay_probability=args.control_delay_probability,
        disturbance_reset_joint_noise=args.disturbance_reset_joint_noise,
        disturbance_reset_height_noise=args.disturbance_reset_height_noise,
        recovery_window_steps=args.recovery_window_steps,
        recovery_velocity_ema_alpha=args.recovery_velocity_ema_alpha,
        recovery_forward_tolerance=args.recovery_forward_tolerance,
        recovery_lateral_tolerance=args.recovery_lateral_tolerance,
        recovery_required_steps=args.recovery_required_steps,
        residual_scale=tuple(
            value * args.residual_scale_multiplier
            for value in ForwardTeacherStudentConfig().residual_scale
        ),
    )
    reference_spec, reference_source = _resolve_reference_spec(args)
    print(
        "stage=ik_reference status=building source=xml_stand "
        f"speed_mode={reference_source['mode']} command_vx={args.command_vx:g} "
        f"frequency={reference_spec.frequency:g} stride={reference_spec.stride_length:g} "
        f"height={reference_spec.step_height:g} duty={reference_spec.duty:g}",
        flush=True,
    )
    reference = build_ik_reference(args.xml_path, reference_spec)
    np.savez(
        args.out / "ik_reference.npz",
        joint_targets=reference.joint_targets,
        desired_contacts=reference.desired_contacts,
        stand_q=reference.stand_q,
    )
    run_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    run_config["resolved_ik_reference"] = asdict(reference_spec)
    run_config["ik_reference_source"] = reference_source
    run_config["teacher_step_plan"] = training_plan
    (args.out / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    nominal_config = replace(config, disturbance_enabled=False)
    teacher_env = make_forward_teacher_student_env(
        "teacher", config=config, reference=reference, xml_path=args.xml_path, seed=args.seed
    )
    teacher_eval_env = make_forward_teacher_student_env(
        "teacher", config=config, reference=reference, xml_path=args.xml_path, seed=args.seed + 10_000
    )
    nominal_eval_env = make_forward_teacher_student_env(
        "teacher",
        config=nominal_config,
        reference=reference,
        xml_path=args.xml_path,
        seed=args.seed + 10_000,
    )
    zero_teacher_policy = lambda obs, rng: (
        jp.zeros(obs.shape[:-1] + (config.action_size,)),
        {},
    )
    nominal_baseline_report = _evaluate_teacher(
        jax,
        nominal_eval_env,
        zero_teacher_policy,
        args.seed + 12_000,
        min(args.teacher_eval_envs, 64),
        args.episode_length,
    )
    (teacher_dir / "ik_baseline_nominal_evaluation.json").write_text(
        json.dumps(nominal_baseline_report, indent=2), encoding="utf-8"
    )
    _print_evaluation_summary("ik_baseline", nominal_baseline_report, "nominal")
    if args.teacher_disturbances:
        disturbed_baseline_report = _evaluate_teacher(
            jax,
            teacher_eval_env,
            zero_teacher_policy,
            args.seed + 13_000,
            min(args.teacher_eval_envs, 64),
            args.episode_length,
        )
        (teacher_dir / "ik_baseline_disturbed_evaluation.json").write_text(
            json.dumps(disturbed_baseline_report, indent=2), encoding="utf-8"
        )
        _print_evaluation_summary("ik_baseline", disturbed_baseline_report, "disturbed")
    else:
        disturbed_baseline_report = nominal_baseline_report
    (teacher_dir / "ik_baseline_evaluation.json").write_text(
        json.dumps(disturbed_baseline_report, indent=2), encoding="utf-8"
    )
    checkpoint_kwargs = {}
    ppo_parameters = inspect.signature(ppo.train).parameters
    if "save_checkpoint_path" in ppo_parameters:
        checkpoint_kwargs["save_checkpoint_path"] = str((teacher_dir / "ppo_checkpoint").resolve())
    if args.teacher_restore is not None:
        if "restore_checkpoint_path" not in ppo_parameters:
            raise SystemExit("The installed Brax version cannot restore PPO checkpoints")
        checkpoint_kwargs["restore_checkpoint_path"] = str(
            _resolve_latest_checkpoint(args.teacher_restore)
        )

    best_teacher = {
        "score": None,
        "step": 0,
        "params": None,
        "summary": {},
        "summaries_by_step": {},
        "pending_params": {},
    }

    def consider_teacher(step, params, summary):
        reward_per_step = summary.get("reward_per_step")
        if reward_per_step is None:
            return
        score = _teacher_selection_score(summary)
        if best_teacher["score"] is None or score > best_teacher["score"]:
            best_teacher.update(
                score=score,
                step=int(step),
                params=params,
                summary=summary,
            )
            model_io.save_params(teacher_dir / "params_ppo_best", params)
            print(
                f"stage=teacher_best step={int(step):,} score={score:.5f} "
                f"vx={summary.get('mean_velocity_x', float('nan')):.4f} "
                f"vx_error={summary.get('mean_velocity_error', float('nan')):.4f} "
                f"roll_pitch={summary.get('mean_roll_pitch_rate_rms', float('nan')):.4f} "
                f"failure={summary.get('failure_rate', 0.0):.3f}",
                flush=True,
            )

    def progress_fn(step, metrics):
        step = int(step)
        _teacher_progress(step, metrics, config.command_vx)
        summary = _teacher_eval_summary(metrics, config.command_vx)
        best_teacher["summaries_by_step"][step] = summary
        params = best_teacher["pending_params"].pop(step, None)
        if params is not None:
            consider_teacher(step, params, summary)

    def policy_params_fn(step, make_policy, params):
        del make_policy
        step = int(step)
        summary = best_teacher["summaries_by_step"].get(step)
        if summary is None:
            best_teacher["pending_params"][step] = params
        else:
            consider_teacher(step, params, summary)

    if "policy_params_fn" in ppo_parameters:
        checkpoint_kwargs["policy_params_fn"] = policy_params_fn
    else:
        print("stage=teacher_best status=unsupported_by_installed_brax", flush=True)

    print(
        f"stage=teacher_train steps={args.teacher_steps:,} envs={args.teacher_envs} "
        f"obs={teacher_env.observation_size} action={teacher_env.action_size}",
        flush=True,
    )
    make_teacher_policy, teacher_params, teacher_metrics = ppo.train(
        environment=teacher_env,
        eval_env=teacher_eval_env,
        num_timesteps=args.teacher_steps,
        episode_length=args.episode_length,
        action_repeat=1,
        num_envs=args.teacher_envs,
        num_evals=args.teacher_evals,
        num_eval_envs=args.teacher_eval_envs,
        learning_rate=args.teacher_learning_rate,
        entropy_cost=args.teacher_entropy_cost,
        discounting=0.99,
        reward_scaling=1.0,
        unroll_length=args.teacher_unroll_length,
        batch_size=args.teacher_batch_size,
        num_minibatches=args.teacher_minibatches,
        num_updates_per_batch=args.teacher_updates_per_batch,
        normalize_observations=True,
        network_factory=make_network_factory(
            args.teacher_hidden,
            "elu",
            zero_policy_output=args.teacher_zero_policy_init,
        ),
        progress_fn=progress_fn,
        seed=args.seed,
        **checkpoint_kwargs,
    )
    model_io.save_params(teacher_dir / "params_final", teacher_params)
    ppo_teacher_params = (
        best_teacher["params"] if best_teacher["params"] is not None else teacher_params
    )
    ppo_selected_step = (
        best_teacher["step"] if best_teacher["params"] is not None else args.teacher_steps
    )
    ppo_teacher_policy = make_teacher_policy(ppo_teacher_params, deterministic=True)
    nominal_ppo_report = _evaluate_teacher(
        jax,
        nominal_eval_env,
        ppo_teacher_policy,
        args.seed + 15_000,
        args.eval_envs,
        args.episode_length,
    )
    nominal_baseline_selection_report = _evaluate_teacher(
        jax,
        nominal_eval_env,
        zero_teacher_policy,
        args.seed + 15_000,
        args.eval_envs,
        args.episode_length,
    )
    if args.teacher_disturbances:
        disturbed_ppo_report = _evaluate_teacher(
            jax,
            teacher_eval_env,
            ppo_teacher_policy,
            args.seed + 16_000,
            args.eval_envs,
            args.episode_length,
        )
        disturbed_baseline_selection_report = _evaluate_teacher(
            jax,
            teacher_eval_env,
            zero_teacher_policy,
            args.seed + 16_000,
            args.eval_envs,
            args.episode_length,
        )
    else:
        disturbed_ppo_report = nominal_ppo_report
        disturbed_baseline_selection_report = nominal_baseline_selection_report

    ppo_preserves_baseline = _ppo_preserves_baseline(
        nominal_ppo_report, nominal_baseline_selection_report
    )
    ppo_improves_disturbed = _ppo_improves_disturbed_baseline(
        disturbed_ppo_report,
        disturbed_baseline_selection_report,
        args.min_disturbed_score_improvement,
    )
    if args.teacher_selection_mode == "robust":
        ppo_score = _disturbed_teacher_score(disturbed_ppo_report)
        baseline_score = _disturbed_teacher_score(disturbed_baseline_selection_report)
        select_ppo = ppo_preserves_baseline and ppo_improves_disturbed
    else:
        ppo_score = _teacher_selection_score(nominal_ppo_report)
        baseline_score = _teacher_selection_score(nominal_baseline_selection_report)
        select_ppo = _should_select_ppo(
            args.teacher_selection_mode,
            ppo_score,
            baseline_score,
            ppo_preserves_baseline,
        )
    if not select_ppo:
        selected_source = "ik_baseline"
        selected_step = 0
        teacher_policy = zero_teacher_policy
        teacher_report = dict(nominal_baseline_selection_report)
        selected_params_path = None
    else:
        selected_source = "ppo"
        selected_step = ppo_selected_step
        teacher_policy = ppo_teacher_policy
        teacher_report = dict(nominal_ppo_report)
        selected_params_path = teacher_dir / "params"
        model_io.save_params(selected_params_path, ppo_teacher_params)

    selection = {
        "selected_source": selected_source,
        "selected_step": selected_step,
        "selected_score": ppo_score if selected_source == "ppo" else baseline_score,
        "ppo_step": ppo_selected_step,
        "ppo_score": ppo_score,
        "ppo_preserves_baseline": ppo_preserves_baseline,
        "ppo_improves_disturbed_baseline": ppo_improves_disturbed,
        "selection_mode": args.teacher_selection_mode,
        "ppo_report": nominal_ppo_report,
        "ppo_nominal_report": nominal_ppo_report,
        "ppo_disturbed_report": disturbed_ppo_report,
        "ik_baseline_score": baseline_score,
        "ik_baseline_report": nominal_baseline_selection_report,
        "ik_nominal_baseline_report": nominal_baseline_selection_report,
        "ik_disturbed_baseline_report": disturbed_baseline_selection_report,
    }
    (teacher_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    (teacher_dir / "ppo_evaluation.json").write_text(
        json.dumps(nominal_ppo_report, indent=2), encoding="utf-8"
    )
    (teacher_dir / "ppo_disturbed_evaluation.json").write_text(
        json.dumps(disturbed_ppo_report, indent=2), encoding="utf-8"
    )
    print(
        f"stage=teacher_selection source={selected_source} selected_step={selected_step:,} "
        f"ppo_score={ppo_score:.5f} ik_baseline_score={baseline_score:.5f} "
        f"nominal_preserved={ppo_preserves_baseline} "
        f"disturbed_improved={ppo_improves_disturbed} "
        f"saved={selected_params_path}",
        flush=True,
    )
    _print_teacher_comparison(
        "nominal", nominal_ppo_report, nominal_baseline_selection_report
    )
    if args.teacher_disturbances:
        _print_teacher_comparison(
            "disturbed",
            disturbed_ppo_report,
            disturbed_baseline_selection_report,
            _disturbed_teacher_score(disturbed_ppo_report)
            - _disturbed_teacher_score(disturbed_baseline_selection_report),
        )

    absolute_teacher_accepted = (
        teacher_report["mean_velocity_x"] >= args.min_accepted_teacher_vx
        and teacher_report["failure_rate"] <= args.max_accepted_teacher_failure_rate
        and teacher_report["mean_velocity_error"]
        <= args.max_accepted_teacher_velocity_error
        and teacher_report["mean_roll_pitch_rate_rms"]
        <= args.max_accepted_teacher_roll_pitch_rate
        and teacher_report["mean_abs_velocity_y"]
        <= args.max_accepted_teacher_lateral_speed
        and teacher_report["mean_abs_yaw_rate"]
        <= args.max_accepted_teacher_yaw_rate
    )
    teacher_accepted = _teacher_gate_acceptance(
        args.teacher_selection_mode,
        selected_source,
        absolute_teacher_accepted,
        ppo_preserves_baseline,
        ppo_improves_disturbed,
        args.allow_ik_baseline_teacher,
    )

    if not teacher_accepted:
        if args.teacher_selection_mode == "preserve":
            teacher_report["rejection_reason"] = "ppo_teacher_did_not_preserve_ik_baseline"
        elif args.teacher_selection_mode == "robust":
            teacher_report["rejection_reason"] = "ppo_teacher_failed_nominal_or_disturbed_gate"
        elif selected_source != "ppo" and not args.allow_ik_baseline_teacher:
            teacher_report["rejection_reason"] = "ppo_teacher_did_not_outperform_ik_baseline"
    teacher_report["accepted"] = teacher_accepted
    teacher_report["absolute_thresholds_passed"] = absolute_teacher_accepted
    teacher_report["ppo_preserves_nominal_baseline"] = ppo_preserves_baseline
    teacher_report["ppo_improves_disturbed_baseline"] = ppo_improves_disturbed
    teacher_report["nominal_evaluation"] = (
        nominal_ppo_report if selected_source == "ppo" else nominal_baseline_selection_report
    )
    teacher_report["disturbed_evaluation"] = (
        disturbed_ppo_report
        if selected_source == "ppo"
        else disturbed_baseline_selection_report
    )
    teacher_report["minimum_velocity_x"] = args.min_accepted_teacher_vx
    teacher_report["maximum_failure_rate"] = args.max_accepted_teacher_failure_rate
    teacher_report["maximum_velocity_error"] = args.max_accepted_teacher_velocity_error
    teacher_report["maximum_roll_pitch_rate_rms"] = (
        args.max_accepted_teacher_roll_pitch_rate
    )
    teacher_report["maximum_abs_velocity_y"] = args.max_accepted_teacher_lateral_speed
    teacher_report["maximum_abs_yaw_rate"] = args.max_accepted_teacher_yaw_rate
    teacher_report["selected_source"] = selected_source
    teacher_report["selected_step"] = selected_step
    (teacher_dir / "evaluation.json").write_text(
        json.dumps(teacher_report, indent=2), encoding="utf-8"
    )
    selected_nominal_report = (
        nominal_ppo_report if selected_source == "ppo" else nominal_baseline_selection_report
    )
    selected_disturbed_report = (
        disturbed_ppo_report
        if selected_source == "ppo"
        else disturbed_baseline_selection_report
    )
    _print_evaluation_summary("teacher_result", selected_nominal_report, "nominal")
    if args.teacher_disturbances:
        _print_evaluation_summary("teacher_result", selected_disturbed_report, "disturbed")
    print(
        f"stage=teacher_acceptance accepted={teacher_accepted} "
        f"source={selected_source} step={selected_step:,} "
        f"nominal_preserved={ppo_preserves_baseline} "
        f"disturbed_improved={ppo_improves_disturbed} "
        f"absolute_thresholds={absolute_teacher_accepted} "
        f"report={teacher_dir / 'evaluation.json'}",
        flush=True,
    )
    if args.strict_acceptance and not teacher_accepted:
        print("stage=pipeline_stopped reason=teacher_acceptance_failed", flush=True)
        raise SystemExit(2)
    if args.teacher_only:
        print(
            f"stage=pipeline_done mode=teacher_only accepted={teacher_accepted} "
            f"source={selected_source}",
            flush=True,
        )
        return

    observations, labels = _collect_teacher_dataset(
        jax,
        jp,
        teacher_env,
        teacher_policy,
        args.seed + 20_000,
        args.rollout_envs,
        args.rollout_horizon,
        args.dataset_samples,
    )
    obs_mean = jp.asarray(np.mean(observations, axis=0).astype(np.float32))
    obs_std = jp.asarray(np.maximum(np.std(observations, axis=0), 1e-3).astype(np.float32))
    layer_sizes = [config.student_observation_size, *args.student_hidden, config.action_size]
    student_params = _student_init(jax, jp, jax.random.PRNGKey(args.seed + 30_000), layer_sizes)
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
        args.seed + 40_000,
        "student_bc",
    )

    dagger_env = make_forward_teacher_student_env(
        "dagger", config=config, reference=reference, xml_path=args.xml_path, seed=args.seed + 50_000
    )
    all_observations = observations
    all_labels = labels
    for round_index in range(args.dagger_rounds):
        dagger_obs, dagger_labels = _collect_dagger_dataset(
            jax,
            jp,
            dagger_env,
            teacher_policy,
            student_params,
            obs_mean,
            obs_std,
            args.seed + 60_000 + 1_000 * round_index,
            args.rollout_envs,
            args.rollout_horizon,
            args.dagger_samples,
        )
        all_observations = np.concatenate((all_observations, dagger_obs))
        all_labels = np.concatenate((all_labels, dagger_labels))
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
            args.seed + 70_000 + round_index,
            f"student_dagger_{round_index + 1}",
        )

    if args.save_dataset:
        np.savez_compressed(
            args.out / "student_dataset.npz",
            observations=all_observations.astype(np.float32),
            actions=all_labels.astype(np.float32),
        )

    student_env = make_forward_teacher_student_env(
        "student", config=config, xml_path=args.xml_path, seed=args.seed + 80_000
    )
    report = _evaluate_student(
        jax,
        jp,
        student_env,
        student_params,
        obs_mean,
        obs_std,
        args.seed + 80_000,
        args.eval_envs,
        args.episode_length,
    )
    accepted = (
        report["mean_velocity_x"] >= args.min_accepted_vx
        and report["failure_rate"] <= args.max_accepted_failure_rate
        and report["mean_velocity_error"] <= args.max_accepted_velocity_error
        and report["mean_roll_pitch_rate_rms"] <= args.max_accepted_roll_pitch_rate
        and report["mean_abs_velocity_y"] <= args.max_accepted_lateral_speed
        and report["mean_abs_yaw_rate"] <= args.max_accepted_yaw_rate
    )
    report["accepted"] = accepted
    report["minimum_velocity_x"] = args.min_accepted_vx
    report["maximum_failure_rate"] = args.max_accepted_failure_rate
    report["maximum_velocity_error"] = args.max_accepted_velocity_error
    report["maximum_roll_pitch_rate_rms"] = args.max_accepted_roll_pitch_rate
    report["maximum_abs_velocity_y"] = args.max_accepted_lateral_speed
    report["maximum_abs_yaw_rate"] = args.max_accepted_yaw_rate

    metadata = {
        "format": "disk_robot_student_mlp_v1",
        "xml_path": str(args.xml_path),
        "stand_source": "xml:keyframe:stand",
        "observation_size": config.student_observation_size,
        "action_size": config.action_size,
        "hidden_layers": args.student_hidden,
        "action_semantics": "q_target = q_stand + student_action_scale * tanh(policy)",
        "student_action_scale": list(config.student_action_scale),
        "command": [config.command_vx, 0.0, 0.0],
        "config": asdict(config),
        "ik_reference": asdict(reference_spec),
        "ik_reference_source": reference_source,
        "teacher_source": selected_source,
        "teacher_selected_step": selected_step,
        "evaluation": report,
    }
    student_path = args.out / "student_policy.npz"
    _save_student_policy(student_path, student_params, obs_mean, obs_std, metadata)
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_evaluation_summary("student_eval", report)
    print(f"stage=pipeline_done student_policy={student_path} accepted={accepted}", flush=True)
    if args.strict_acceptance and not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
