from __future__ import annotations

import argparse

import numpy as np

from disk_robot.walk_config import WalkTaskConfig
from disk_robot.walk_env import DiskRobotWalkEnv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a local MuJoCo smoke test for the disk robot walk task.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("zero", "random"), default="zero")
    parser.add_argument("--command-velocity", type=float, default=0.45)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = WalkTaskConfig(command_velocity=args.command_velocity)
    env = DiskRobotWalkEnv(config=config, seed=args.seed)
    obs, info = env.reset()
    rng = np.random.default_rng(args.seed)
    total_reward = 0.0
    last_info = info
    for _ in range(args.steps):
        if args.policy == "random":
            action = rng.uniform(-config.action_scale, config.action_scale, size=config.action_size)
        else:
            action = np.zeros(config.action_size)
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(
        "walk_smoke "
        f"obs_size={obs.shape[0]} action_size={config.action_size} "
        f"steps={last_info['step_count']} total_reward={total_reward:.3f} "
        f"torso_height={last_info['torso_height']:.3f} "
        f"foot_contacts={last_info['foot_contact_count']} "
        f"disk_contacts={last_info['disk_contact_count']} "
        f"forward_velocity={last_info['forward_velocity']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

