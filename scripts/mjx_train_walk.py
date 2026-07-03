from __future__ import annotations

import argparse
from pathlib import Path

from disk_robot.walk_config import WalkTaskConfig
from disk_robot_mjx.pipeline import configure_cloud_runtime, hidden_layers_tuple, make_network_factory


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
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 128, 128])
    parser.add_argument("--activation", default="elu", choices=["relu", "tanh", "elu", "swish", "silu"])
    parser.add_argument("--xla-triton", action="store_true", default=True)
    parser.add_argument("--no-xla-triton", dest="xla_triton", action="store_false")
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--matmul-precision", default="high")
    parser.add_argument("--runtime-diagnostics", action="store_true", default=True)
    parser.add_argument("--no-runtime-diagnostics", dest="runtime_diagnostics", action="store_false")
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

    config = WalkTaskConfig(command_velocity=args.command_velocity)
    env = make_brax_env(config=config, seed=args.seed, settle_steps=args.settle_steps)
    eval_env = make_brax_env(config=config, seed=args.seed + 10_000, settle_steps=args.settle_steps)
    args.out.mkdir(parents=True, exist_ok=True)
    make_inference_fn, params, _metrics = ppo.train(
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
        seed=args.seed,
    )
    del make_inference_fn
    model_io.save_params(args.out / "params", params)
    print(f"stage=train_done saved={args.out / 'params'}", flush=True)


if __name__ == "__main__":
    main()
