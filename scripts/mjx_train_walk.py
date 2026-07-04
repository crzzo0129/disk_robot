from __future__ import annotations

import argparse
from pathlib import Path

from disk_robot.walk_config import WalkTaskConfig
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
    if "eval/episode_forward_velocity" in logged:
        logged["eval/avg_forward_velocity"] = logged["eval/episode_forward_velocity"] / episode_length
    if "eval/episode_torso_height" in logged:
        logged["eval/avg_torso_height"] = logged["eval/episode_torso_height"] / episode_length


def _make_progress_fn(wandb_run=None):
    def progress(step, metrics):
        logged = {}
        for key, value in metrics.items():
            scalar = _metric_value(value)
            if scalar is not None:
                logged[key] = scalar
        _add_average_eval_metrics(logged)
        if logged:
            print(
                "stage=progress "
                f"step={step} "
                f"eval_reward={logged.get('eval/episode_reward', float('nan')):.3f} "
                f"avg_forward_velocity={logged.get('eval/avg_forward_velocity', float('nan')):.3f} "
                f"avg_torso_height={logged.get('eval/avg_torso_height', float('nan')):.3f} "
                f"sps={logged.get('training/sps', float('nan')):.1f}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(logged, step=int(step))

    return progress


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


def _render_policy_video(args, make_inference_fn, params, config):
    try:
        import imageio.v3 as iio
        import jax
        import mujoco
        import numpy as np
        import wandb
        from disk_robot_mjx.brax_env import make_brax_env
    except ImportError as exc:
        print(f"stage=wandb_video_skipped reason=missing_dependency detail={exc}", flush=True)
        return None

    env = make_brax_env(config=config, seed=args.seed + 20_000, settle_steps=args.settle_steps, xml_path=args.xml_path)
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(args.seed + 30_000)
    state = jit_reset(rng)
    pipeline_states = [jax.device_get(state.pipeline_state)]
    for _ in range(args.wandb_video_steps):
        rng, action_key = jax.random.split(rng)
        action, _ = jit_inference_fn(state.obs, action_key)
        state = jit_step(state, action)
        pipeline_states.append(jax.device_get(state.pipeline_state))
        if bool(jax.device_get(state.done)):
            break

    model = mujoco.MjModel.from_xml_path(str(args.xml_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = args.wandb_video_camera if args.wandb_video_camera else None
    frames = []
    for pipeline_state in pipeline_states:
        qpos = np.asarray(pipeline_state.qpos)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        try:
            renderer.update_scene(data, camera=camera)
        except ValueError:
            renderer.update_scene(data)
        frames.append(renderer.render())
    renderer.close()

    video_path = args.out / "final_policy.mp4"
    iio.imwrite(video_path, frames, fps=args.wandb_video_fps)
    print(f"stage=wandb_video_done saved={video_path} frames={len(frames)}", flush=True)
    return video_path, wandb.Video(str(video_path), format="mp4")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Cloud MJX/Brax PPO training entrypoint for disk robot walking.")
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
    parser.add_argument("--discounting", type=float, default=0.97)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--command-velocity", type=float, default=0.45)
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
    parser.add_argument("--wandb-video-steps", type=int, default=240)
    parser.add_argument("--wandb-video-fps", type=int, default=50)
    parser.add_argument("--wandb-video-camera", default="side_cam")
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
    config = WalkTaskConfig(command_velocity=args.command_velocity)
    print(f"stage=env_config xml_path={args.xml_path}", flush=True)
    env = make_brax_env(config=config, seed=args.seed, settle_steps=args.settle_steps, xml_path=args.xml_path)
    eval_env = make_brax_env(
        config=config,
        seed=args.seed + 10_000,
        settle_steps=args.settle_steps,
        xml_path=args.xml_path,
    )
    args.out.mkdir(parents=True, exist_ok=True)
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
        progress_fn=_make_progress_fn(wandb_run),
        seed=args.seed,
    )
    model_io.save_params(args.out / "params", params)
    print(f"stage=train_done saved={args.out / 'params'}", flush=True)
    if wandb_run is not None:
        wandb_run.summary["params_path"] = str(args.out / "params")
        final_metrics = {}
        for key, value in metrics.items():
            scalar = _metric_value(value)
            if scalar is not None:
                final_metrics[f"final/{key}"] = scalar
        if final_metrics:
            wandb_run.log(final_metrics)
        if args.wandb_video:
            video_result = _render_policy_video(args, make_inference_fn, params, config)
            if video_result is not None:
                video_path, video = video_result
                try:
                    wandb_run.log({"media/final_policy_video": video}, step=int(args.steps))
                    wandb_run.save(str(video_path), base_path=str(args.out), policy="now")
                except Exception as exc:  # noqa: BLE001
                    print(f"stage=wandb_video_upload_failed detail={exc}", flush=True)
        wandb_run.finish()


if __name__ == "__main__":
    main()
