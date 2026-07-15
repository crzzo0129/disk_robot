from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

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
    parser.add_argument("--teacher-learning-rate", type=float, default=3e-4)
    parser.add_argument("--teacher-unroll-length", type=int, default=20)
    parser.add_argument("--teacher-batch-size", type=int, default=256)
    parser.add_argument("--teacher-minibatches", type=int, default=32)
    parser.add_argument("--teacher-updates-per-batch", type=int, default=4)
    parser.add_argument("--teacher-restore", type=Path, default=None)
    parser.add_argument("--min-accepted-teacher-vx", type=float, default=0.04)
    parser.add_argument("--max-accepted-teacher-failure-rate", type=float, default=0.10)

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
    parser.add_argument("--min-accepted-vx", type=float, default=0.04)
    parser.add_argument("--max-accepted-failure-rate", type=float, default=0.10)
    parser.add_argument("--strict-acceptance", action="store_true")

    parser.add_argument("--ik-samples", type=int, default=256)
    parser.add_argument("--ik-frequency", type=float, default=0.8)
    parser.add_argument("--ik-stride", type=float, default=0.04)
    parser.add_argument("--ik-height", type=float, default=0.025)
    parser.add_argument("--ik-duty", type=float, default=0.72)
    parser.add_argument("--command-vx", type=float, default=0.10)
    parser.add_argument("--kp", type=float, default=7.5)
    parser.add_argument("--kd", type=float, default=0.25)
    parser.add_argument("--torque-limit", type=float, default=3.0)
    parser.add_argument("--startup-steps", type=int, default=25)

    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--no-xla-triton", dest="xla_triton", action="store_false")
    parser.set_defaults(xla_triton=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


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


def _teacher_progress(step, metrics):
    names = (
        "eval/episode_reward",
        "eval/avg_episode_length",
        "eval/episode_velocity_x",
        "eval/episode_velocity_error",
        "eval/episode_roll_pitch_rate_rms",
        "eval/episode_failed",
    )
    values = []
    for name in names:
        if name in metrics:
            value = _metric_scalar(metrics[name])
            if value is not None:
                values.append(f"{name.split('/')[-1]}={value:.4f}")
    print(f"stage=teacher_eval step={int(step):,} " + " ".join(values), flush=True)


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


def _collect_teacher_dataset(
    jax,
    jp,
    env,
    teacher_policy,
    seed,
    env_count,
    horizon,
    requested_samples,
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
                current_state.info["student_obs"],
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
        obs = obs[valid]
        target = target[valid]
        take = min(len(obs), requested_samples - collected)
        observations.append(obs[:take])
        labels.append(target[:take])
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
            next_state.metrics["velocity_x"],
            next_state.metrics["velocity_error"],
            next_state.metrics["roll_pitch_rate_rms"],
            next_state.metrics["failed"],
            alive,
        )

    (_, _), values = jax.lax.scan(
        eval_step,
        (state, jax.random.PRNGKey(seed + 1)),
        (),
        length=horizon,
    )
    vx, velocity_error, roll_pitch_rate, failed, alive = [
        np.asarray(jax.device_get(value)) for value in values
    ]
    denominator = max(float(np.sum(alive)), 1.0)
    return {
        "mean_velocity_x": float(np.sum(vx * alive) / denominator),
        "mean_velocity_error": float(np.sum(velocity_error * alive) / denominator),
        "mean_roll_pitch_rate_rms": float(np.sum(roll_pitch_rate * alive) / denominator),
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
):
    obs = jp.asarray(observations)
    target = jp.asarray(labels)
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(params)

    def loss_fn(current_params, batch_obs, batch_target):
        prediction = _normalized_student_apply(jp, current_params, batch_obs, obs_mean, obs_std)
        return jp.mean(jp.square(prediction - batch_target))

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
            valid = 1.0 - current_state.done
            next_state = step_batch(current_state, student_action)
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
        obs = obs[valid]
        target = target[valid]
        take = min(len(obs), requested_samples - collected)
        observations.append(obs[:take])
        labels.append(target[:take])
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
            next_state.metrics["velocity_x"],
            next_state.metrics["velocity_error"],
            next_state.metrics["roll_pitch_rate_rms"],
            next_state.metrics["disk_contact_count"],
            next_state.metrics["failed"],
            alive,
        )

    _, values = jax.lax.scan(eval_step, state, (), length=horizon)
    vx, velocity_error, roll_pitch_rate, disk_contact, failed, alive = [
        np.asarray(jax.device_get(value)) for value in values
    ]
    denominator = max(float(np.sum(alive)), 1.0)
    failed_any = np.max(failed, axis=0)
    return {
        "mean_velocity_x": float(np.sum(vx * alive) / denominator),
        "mean_velocity_error": float(np.sum(velocity_error * alive) / denominator),
        "mean_roll_pitch_rate_rms": float(np.sum(roll_pitch_rate * alive) / denominator),
        "mean_disk_contacts": float(np.sum(disk_contact * alive) / denominator),
        "failure_rate": float(np.mean(failed_any)),
        "mean_alive_steps": float(np.mean(np.sum(alive, axis=0))),
    }


def main(argv=None):
    args = parse_args(argv)
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
    )
    reference_spec = IKReferenceSpec(
        samples=args.ik_samples,
        frequency=args.ik_frequency,
        stride_length=args.ik_stride,
        step_height=args.ik_height,
        duty=args.ik_duty,
        mode="trot",
    )
    print("stage=ik_reference status=building source=xml_stand", flush=True)
    reference = build_ik_reference(args.xml_path, reference_spec)
    np.savez(
        args.out / "ik_reference.npz",
        joint_targets=reference.joint_targets,
        desired_contacts=reference.desired_contacts,
        stand_q=reference.stand_q,
    )

    teacher_env = make_forward_teacher_student_env(
        "teacher", config=config, reference=reference, xml_path=args.xml_path, seed=args.seed
    )
    teacher_eval_env = make_forward_teacher_student_env(
        "teacher", config=config, reference=reference, xml_path=args.xml_path, seed=args.seed + 10_000
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
        entropy_cost=5e-3,
        discounting=0.99,
        reward_scaling=1.0,
        unroll_length=args.teacher_unroll_length,
        batch_size=args.teacher_batch_size,
        num_minibatches=args.teacher_minibatches,
        num_updates_per_batch=args.teacher_updates_per_batch,
        normalize_observations=True,
        network_factory=make_network_factory(args.teacher_hidden, "elu"),
        progress_fn=_teacher_progress,
        seed=args.seed,
        **checkpoint_kwargs,
    )
    model_io.save_params(teacher_dir / "params", teacher_params)
    teacher_policy = make_teacher_policy(teacher_params, deterministic=True)
    print(f"stage=teacher_train status=done saved={teacher_dir / 'params'}", flush=True)

    teacher_report = _evaluate_teacher(
        jax,
        teacher_env,
        teacher_policy,
        args.seed + 15_000,
        args.eval_envs,
        args.episode_length,
    )
    teacher_accepted = (
        teacher_report["mean_velocity_x"] >= args.min_accepted_teacher_vx
        and teacher_report["failure_rate"] <= args.max_accepted_teacher_failure_rate
    )
    teacher_report["accepted"] = teacher_accepted
    teacher_report["minimum_velocity_x"] = args.min_accepted_teacher_vx
    teacher_report["maximum_failure_rate"] = args.max_accepted_teacher_failure_rate
    (teacher_dir / "evaluation.json").write_text(
        json.dumps(teacher_report, indent=2), encoding="utf-8"
    )
    print(f"stage=teacher_acceptance {json.dumps(teacher_report, sort_keys=True)}", flush=True)
    if args.strict_acceptance and not teacher_accepted:
        print("stage=pipeline_stopped reason=teacher_acceptance_failed", flush=True)
        raise SystemExit(2)

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
    )
    report["accepted"] = accepted
    report["minimum_velocity_x"] = args.min_accepted_vx
    report["maximum_failure_rate"] = args.max_accepted_failure_rate

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
        "evaluation": report,
    }
    student_path = args.out / "student_policy.npz"
    _save_student_policy(student_path, student_params, obs_mean, obs_std, metadata)
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "run_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2),
        encoding="utf-8",
    )
    print(f"stage=student_eval {json.dumps(report, sort_keys=True)}", flush=True)
    print(f"stage=pipeline_done student_policy={student_path} accepted={accepted}", flush=True)
    if args.strict_acceptance and not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
