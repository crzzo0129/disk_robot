from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from disk_robot.walk_config import command_profile
from disk_robot.walk_env import DEFAULT_XML, DiskRobotWalkEnv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the gait-free CPU MuJoCo walk task.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("zero", "random"), default="zero")
    parser.add_argument("--command-profile", choices=("forward", "omni", "full"), default="forward")
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--reset-joint-noise", type=float, default=None)
    parser.add_argument("--reset-height-noise", type=float, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = command_profile(args.command_profile)
    overrides = {}
    if args.reset_joint_noise is not None:
        overrides["reset_joint_noise"] = args.reset_joint_noise
    if args.reset_height_noise is not None:
        overrides["reset_height_noise"] = args.reset_height_noise
    if overrides:
        config = replace(config, **overrides)
    env = DiskRobotWalkEnv(config=config, xml_path=args.xml_path, seed=args.seed)
    obs, info = env.reset()
    rng = np.random.default_rng(args.seed)
    total_reward = 0.0
    last_info = info
    for _ in range(args.steps):
        action = (
            rng.uniform(-1.0, 1.0, size=config.action_size)
            if args.policy == "random"
            else np.zeros(config.action_size)
        )
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(
        "walk_smoke "
        f"model={args.xml_path.name} profile={args.command_profile} "
        f"obs_size={obs.shape[0]} action_size={config.action_size} "
        f"steps={last_info['step_count']} total_reward={total_reward:.3f} "
        f"command=({last_info['command_x']:.3f},{last_info['command_y']:.3f},{last_info['command_yaw']:.3f}) "
        f"velocity=({last_info['velocity_x']:.3f},{last_info['velocity_y']:.3f},{last_info['yaw_rate']:.3f}) "
        f"torso_height={last_info['torso_height']:.3f} upright={last_info['upright']:.3f} "
        f"foot_contacts={last_info['foot_contact_count']} disk_contacts={last_info['disk_contact_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
