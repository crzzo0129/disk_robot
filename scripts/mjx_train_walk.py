from __future__ import annotations

import argparse
import inspect
import shutil
from dataclasses import replace
from pathlib import Path

from disk_robot.walk_config import WalkTaskConfig, command_profile
from disk_robot.walk_reward import REWARD_TERM_NAMES
from disk_robot_mjx.brax_env import TRAIN_XML_PATH
from disk_robot_mjx.pipeline import configure_cloud_runtime, hidden_layers_tuple, make_network_factory


def _metric_value(value):
    try:
        import jax
    except ImportError:
        jax = None
    if jax is not None:
        value = jax.device_get(value)
    try:
        if getattr(value, "shape", ()) == ():
            return float(value)
    except (TypeError, ValueError):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_average_eval_metrics(logged):
    episode_length = logged.get("eval/avg_episode_length")
    if episode_length is None or episode_length <= 0:
        return
    for name in (
        "velocity_x",
        "velocity_y",
        "yaw_rate",
        "velocity_error_xy",
        "yaw_rate_error",
        "torso_height",
        "action_rms",
        "action_rate_rms",
    ):
        key = f"eval/episode_{name}"
        if key in logged:
            logged[f"eval/avg_{name}"] = logged[key] / episode_length


def _print_eval_report(step, logged):
    terms = []
    for name in REWARD_TERM_NAMES:
        key = f"eval/episode_reward_{name}"
        if key in logged:
            terms.append((name, logged[key]))
    episode_length = logged.get("eval/avg_episode_length", 0.0)
    episode_reward = logged.get("eval/episode_reward", sum(value for _, value in terms))
    term_sum = sum(value for _, value in terms)
    abs_sum = sum(abs(value) for _, value in terms)
    avg_velocity_error = logged.get("eval/avg_velocity_error_xy", float("nan"))
    avg_yaw_error = logged.get("eval/avg_yaw_rate_error", float("nan"))
    avg_torso_height = logged.get("eval/avg_torso_height", float("nan"))
    failed = logged.get("eval/episode_failed", float("nan"))
    height_failed = logged.get("eval/episode_height_failed", float("nan"))
    upright_failed = logged.get("eval/episode_upright_failed", float("nan"))
    timeout = logged.get("eval/episode_timeout", float("nan"))
    sps = logged.get("training/sps", float("nan"))

    width = 112
    print("\n" + "=" * width)
    print(
        f"EVAL step={int(step):,} | "
        f"reward={episode_reward:.3f} | "
        f"len={episode_length:.1f} | "
        f"vel_rmse={avg_velocity_error:.4f} | "
        f"yaw_err={avg_yaw_error:.4f} | "
        f"avg_h={avg_torso_height:.4f} | "
        f"fail={failed:.3f} | "
        f"h_fail={height_failed:.3f} | "
        f"u_fail={upright_failed:.3f} | "
        f"timeout={timeout:.3f} | "
        f"sps={sps:.1f}"
    )
    print(f"reward term sum={term_sum:.3f} | abs term sum={abs_sum:.3f}")
    if not terms:
        print("no eval reward terms found")
        print("=" * width, flush=True)
        return

    print("-" * width)
    print(f"{'term':<20} {'episode':>13} {'per_step':>12} {'abs%':>8} {'total%':>9}")
    print("-" * width)
    for name, value in sorted(terms, key=lambda item: abs(item[1]), reverse=True):
        per_step = value / episode_length if episode_length > 0 else float("nan")
        pct_abs = 100.0 * abs(value) / abs_sum if abs_sum > 0 else 0.0
        pct_total = 100.0 * value / episode_reward if episode_reward else float("nan")
        print(f"{name:<20} {value:>13.3f} {per_step:>12.5f} {pct_abs:>7.2f}% {pct_total:>8.2f}%")
    print("=" * width, flush=True)


def _make_progress_fn(wandb_run=None):
    def progress(step, metrics):
        logged = {}
        for key, value in metrics.items():
            scalar = _metric_value(value)
            if scalar is not None:
                logged[key] = scalar
        _add_average_eval_metrics(logged)
        if logged:
            _print_eval_report(step, logged)
            if wandb_run is not None:
                wandb_run.log(logged, step=int(step))

    return progress


def _best_policy_score(metrics, config, mode="tracking"):
    del config
    logged = {}
    for key, value in metrics.items():
        scalar = _metric_value(value)
        if scalar is not None:
            logged[key] = scalar
    _add_average_eval_metrics(logged)
    episode_length = logged.get("eval/avg_episode_length")
    velocity_error = logged.get("eval/avg_velocity_error_xy")
    yaw_error = logged.get("eval/avg_yaw_rate_error")
    if episode_length is None:
        return None, logged
    if mode == "survival":
        return episode_length, logged
    if mode == "reward_per_step":
        reward = logged.get("eval/episode_reward")
        if reward is None or episode_length <= 0:
            return None, logged
        return reward / episode_length, logged
    if episode_length <= 0 or velocity_error is None or yaw_error is None:
        return None, logged
    return -velocity_error - 0.5 * yaw_error + 0.0005 * episode_length, logged


def _init_wandb(args):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("--use-wandb requires wandb. Install requirements-mjx.txt first.") from exc
    config = vars(args).copy()
    config["xml_path"] = str(args.xml_path)
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=config,
        settings={"_service_wait": 90, "init_timeout": 90},
    )
    return run


def _copy_path(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _resolve_restore_checkpoint(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_dir():
        return path
    numbered = sorted(
        (child for child in path.iterdir() if child.is_dir() and child.name.isdigit()),
        key=lambda child: int(child.name),
    )
    return numbered[-1] if numbered else path


def _save_params_with_alias(model_io, args, params, step: int, alias: str):
    step_path = args.out / f"params_{alias}_{int(step)}"
    alias_path = args.out / f"params_{alias}"
    model_io.save_params(step_path, params)
    _copy_path(step_path, alias_path)
    print(f"checkpoint saved: {alias} step={int(step):,}", flush=True)
    return alias_path


def _render_policy_video(args, make_inference_fn, params, config, name="final_policy"):
    try:
        import imageio.v3 as iio
        import jax
        import mujoco
        import numpy as np
        from disk_robot_mjx.brax_env import make_brax_env
    except ImportError as exc:
        print(f"stage=video_skipped reason=missing_dependency detail={exc}", flush=True)
        return None

    env = make_brax_env(config=config, seed=args.seed + 20_000, settle_steps=args.settle_steps, xml_path=args.xml_path)
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed + 30_000)
    state = jit_reset(rng)
    pipeline_states = [jax.device_get(state.pipeline_state)]
    for _ in range(args.video_steps):
        rng, action_key = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, action_key)
        state = jit_step(state, action)
        pipeline_states.append(jax.device_get(state.pipeline_state))
        if bool(jax.device_get(state.done)):
            break

    model = mujoco.MjModel.from_xml_path(str(args.xml_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.video_height, width=args.video_width)
    camera = args.video_camera if args.video_camera else None
    tracking_camera = mujoco.MjvCamera() if camera == "tracking" else None
    if tracking_camera is not None:
        track_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.video_track_body)
        if track_body_id < 0:
            raise ValueError(f"Tracking body not found: {args.video_track_body}")
        tracking_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        tracking_camera.trackbodyid = track_body_id
        tracking_camera.distance = args.video_distance
        tracking_camera.azimuth = args.video_azimuth
        tracking_camera.elevation = args.video_elevation
    frames = []
    try:
        for pipeline_state in pipeline_states:
            qpos = np.asarray(pipeline_state.qpos)
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            if tracking_camera is not None:
                tracking_camera.lookat[:] = data.xpos[tracking_camera.trackbodyid]
                renderer.update_scene(data, camera=tracking_camera)
            else:
                try:
                    renderer.update_scene(data, camera=camera)
                except ValueError:
                    renderer.update_scene(data)
            frames.append(renderer.render())
    finally:
        renderer.close()

    video_path = args.out / f"{name}.mp4"
    iio.imwrite(video_path, frames, fps=args.video_fps)
    print(f"stage=video_done name={name} saved={video_path} frames={len(frames)}", flush=True)
    return video_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Gait-free command-conditioned MJX PPO training for disk robot.")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--episode-length", type=int, default=128)
    parser.add_argument("--num-evals", type=int, default=5)
    parser.add_argument("--num-eval-envs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--unroll-length", type=int, default=20)
    parser.add_argument("--num-minibatches", type=int, default=32)
    parser.add_argument("--num-updates-per-batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-cost", type=float, default=1e-2)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--command-profile", choices=["forward", "omni", "full"], default="forward")
    parser.add_argument("--command-vx", type=float, nargs=2, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--command-vy", type=float, nargs=2, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--command-yaw", type=float, nargs=2, metavar=("MIN", "MAX"), default=None)
    parser.add_argument("--command-zero-probability", type=float, default=None)
    parser.add_argument("--command-resample-steps", type=int, default=None)
    parser.add_argument("--observation-history", type=int, default=None)
    parser.add_argument("--action-scale", type=float, nargs=12, default=None)
    parser.add_argument("--teacher-blend", type=float, default=0.0)
    parser.add_argument("--reward-teacher-imitation", type=float, default=0.0)
    parser.add_argument("--restore-checkpoint", type=Path, default=None)
    parser.add_argument("--min-torso-height", type=float, default=None)
    parser.add_argument("--terminate-upright", type=float, default=None)
    parser.add_argument("--penalty-termination", type=float, default=None)
    parser.add_argument("--best-score-mode", default="tracking", choices=["tracking", "reward_per_step", "survival"])
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("mjx_runs") / "walk_smoke")
    parser.add_argument("--xml-path", type=Path, default=TRAIN_XML_PATH)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 128, 128])
    parser.add_argument("--activation", default="elu", choices=["relu", "tanh", "elu", "swish", "silu"])
    parser.add_argument("--xla-triton", action="store_true", default=True)
    parser.add_argument("--no-xla-triton", dest="xla_triton", action="store_false")
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--matmul-precision", default="high")
    parser.add_argument("--runtime-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-runtime-diagnostics", dest="runtime_diagnostics", action="store_false")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="disk_robot_walk")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--render-video", action="store_true", default=True)
    parser.add_argument("--no-render-video", dest="render_video", action="store_false")
    parser.add_argument("--video-steps", type=int, default=750)
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--video-camera", default="tracking", help="Fixed camera name, or 'tracking' for a body-following camera.")
    parser.add_argument("--video-track-body", default="base_link")
    parser.add_argument("--video-distance", type=float, default=1.8)
    parser.add_argument("--video-azimuth", type=float, default=90.0)
    parser.add_argument("--video-elevation", type=float, default=-20.0)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--no-wandb-video", dest="wandb_video", action="store_false")
    parser.set_defaults(wandb_video=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_cloud_runtime(
        xla_triton=args.xla_triton,
        mujoco_gl=args.mujoco_gl,
        matmul_precision=args.matmul_precision,
        verbose=args.runtime_diagnostics,
    )
    try:
        from brax.io import model as model_io
        from brax.training.agents.ppo import train as ppo
        from disk_robot_mjx.brax_env import make_brax_env
    except ImportError as exc:
        raise SystemExit(
            "MJX walk training requires the cloud MJX stack. Activate the same "
            "kind of environment used for robot_curl MJX training, with brax, "
            "jax, mujoco, and mujoco-mjx installed."
        ) from exc

    wandb_run = _init_wandb(args)
    config = command_profile(args.command_profile)
    overrides = {}
    if args.action_scale is not None:
        overrides["action_scale"] = tuple(args.action_scale)
    if args.command_vx is not None:
        overrides.update(command_vx_min=args.command_vx[0], command_vx_max=args.command_vx[1])
    if args.command_vy is not None:
        overrides.update(command_vy_min=args.command_vy[0], command_vy_max=args.command_vy[1])
    if args.command_yaw is not None:
        overrides.update(command_yaw_min=args.command_yaw[0], command_yaw_max=args.command_yaw[1])
    if args.command_zero_probability is not None:
        overrides["command_zero_probability"] = args.command_zero_probability
    if args.command_resample_steps is not None:
        overrides["command_resample_steps"] = args.command_resample_steps
    if args.observation_history is not None:
        overrides["observation_history"] = args.observation_history
    if not 0.0 <= args.teacher_blend <= 1.0:
        raise SystemExit("--teacher-blend must be in [0, 1]")
    overrides["teacher_blend"] = args.teacher_blend
    overrides["reward_teacher_imitation"] = args.reward_teacher_imitation
    if args.min_torso_height is not None:
        overrides["min_torso_height"] = args.min_torso_height
    if args.terminate_upright is not None:
        overrides["terminate_upright"] = args.terminate_upright
    if args.penalty_termination is not None:
        overrides["penalty_termination"] = args.penalty_termination
    if args.max_episode_steps is not None:
        overrides["max_episode_steps"] = args.max_episode_steps
    if overrides:
        config = replace(config, **overrides)
    print(
        "training config: "
        f"profile={args.command_profile} "
        f"vx=[{config.command_vx_min},{config.command_vx_max}] "
        f"vy=[{config.command_vy_min},{config.command_vy_max}] "
        f"yaw=[{config.command_yaw_min},{config.command_yaw_max}] "
        f"zero_p={config.command_zero_probability} "
        f"action_scale={config.action_scale} "
        f"history={config.observation_history} "
        f"teacher_blend={config.teacher_blend} "
        f"teacher_reward={config.reward_teacher_imitation} "
        f"min_h={config.min_torso_height} "
        f"term_upright={config.terminate_upright} "
        f"term_penalty={config.penalty_termination} "
        f"best={args.best_score_mode} "
        f"max_steps={config.max_episode_steps}",
        flush=True,
    )
    env = make_brax_env(config=config, seed=args.seed, settle_steps=args.settle_steps, xml_path=args.xml_path)
    eval_env = make_brax_env(
        config=config,
        seed=args.seed + 10_000,
        settle_steps=args.settle_steps,
        xml_path=args.xml_path,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    best_policy = {"score": None, "step": 0, "params": None, "metrics": {}}

    def policy_params_fn(step, make_policy, params):
        del make_policy
        _save_params_with_alias(model_io, args, params, step, "latest")
        if not best_policy["metrics"]:
            return
        score, logged = _best_policy_score(best_policy["metrics"], config, args.best_score_mode)
        if score is None:
            return
        if best_policy["score"] is None or score > best_policy["score"]:
            best_policy.update(score=score, step=int(step), params=params, metrics=logged)
            _save_params_with_alias(model_io, args, params, step, "best")
            print(
                "new best: "
                f"step={int(step):,} "
                f"score={score:.3f} "
                f"vel_err={logged.get('eval/avg_velocity_error_xy', float('nan')):.4f} "
                f"len={logged.get('eval/avg_episode_length', float('nan')):.1f}",
                flush=True,
            )

    def progress_fn(step, metrics):
        best_policy["metrics"] = metrics
        _make_progress_fn(wandb_run)(step, metrics)

    checkpoint_kwargs = {}
    train_parameters = inspect.signature(ppo.train).parameters
    if "save_checkpoint_path" in train_parameters:
        checkpoint_kwargs["save_checkpoint_path"] = str(args.out / "ppo_checkpoint")
    if args.restore_checkpoint is not None:
        if "restore_checkpoint_path" not in train_parameters:
            raise SystemExit("Installed Brax does not support restore_checkpoint_path; upgrade Brax.")
        restore_path = _resolve_restore_checkpoint(args.restore_checkpoint)
        print(f"restoring PPO weights from {restore_path}", flush=True)
        checkpoint_kwargs["restore_checkpoint_path"] = str(restore_path)

    make_inference_fn, params, metrics = ppo.train(
        environment=env,
        eval_env=eval_env,
        num_timesteps=args.steps,
        episode_length=args.episode_length,
        action_repeat=1,
        num_envs=args.envs,
        num_evals=args.num_evals,
        num_eval_envs=args.num_eval_envs,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        reward_scaling=args.reward_scaling,
        unroll_length=args.unroll_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        normalize_observations=True,
        network_factory=make_network_factory(hidden_layers_tuple(args.hidden_layers), args.activation),
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        seed=args.seed,
        **checkpoint_kwargs,
    )

    model_io.save_params(args.out / "params", params)
    _copy_path(args.out / "params", args.out / "params_final")
    print("training done: final params saved", flush=True)
    rendered_videos = []
    if args.render_video:
        final_video = _render_policy_video(args, make_inference_fn, params, config, name="final_policy")
        if final_video is not None:
            rendered_videos.append(("media/final_policy_video", final_video))
        if best_policy["params"] is not None:
            best_video = _render_policy_video(
                args,
                make_inference_fn,
                best_policy["params"],
                config,
                name="best_policy",
            )
            if best_video is not None:
                rendered_videos.append(("media/best_policy_video", best_video))
    if wandb_run is not None:
        wandb_run.summary["params_path"] = str(args.out / "params")
        if best_policy["score"] is not None:
            wandb_run.summary["best_policy_score"] = float(best_policy["score"])
            wandb_run.summary["best_policy_step"] = int(best_policy["step"])
            wandb_run.summary["best_params_path"] = str(args.out / "params_best")
        final_metrics = {}
        for key, value in metrics.items():
            scalar = _metric_value(value)
            if scalar is not None:
                final_metrics[f"final/{key}"] = scalar
        if final_metrics:
            wandb_run.log(final_metrics)
        if args.wandb_video and rendered_videos:
            try:
                import wandb
            except ImportError as exc:
                print(f"stage=wandb_video_upload_skipped reason=missing_dependency detail={exc}", flush=True)
            else:
                for key, video_path in rendered_videos:
                    try:
                        wandb_run.log({key: wandb.Video(str(video_path), format="mp4")})
                        wandb_run.save(str(video_path), base_path=str(args.out), policy="now")
                    except Exception as exc:  # noqa: BLE001
                        print(f"stage=wandb_video_upload_failed key={key} detail={exc}", flush=True)
        wandb_run.finish()


if __name__ == "__main__":
    main()
