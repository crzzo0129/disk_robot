from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from disk_robot.walk_config import WalkTaskConfig
from disk_robot.walk_env import DiskRobotWalkEnv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a local MuJoCo smoke test for the disk robot walk task.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=("zero", "random"), default="zero")
    parser.add_argument("--command-velocity", type=float, default=0.1)
    parser.add_argument("--reset-joint-noise", type=float, default=None)
    parser.add_argument("--reset-height-noise", type=float, default=None)
    parser.add_argument("--gait-time-offset", type=float, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = WalkTaskConfig(command_velocity=args.command_velocity)
    overrides = {}
    if args.reset_joint_noise is not None:
        overrides["reset_joint_noise"] = args.reset_joint_noise
    if args.reset_height_noise is not None:
        overrides["reset_height_noise"] = args.reset_height_noise
    if args.gait_time_offset is not None:
        overrides["gait_time_offset"] = args.gait_time_offset
    if overrides:
        config = replace(config, **overrides)
    env = DiskRobotWalkEnv(config=config, seed=args.seed)
    obs, info = env.reset()
    rng = np.random.default_rng(args.seed)
    total_reward = 0.0
    last_info = info
    start_x = info["torso_x"]
    start_y = info["torso_y"]
    for _ in range(args.steps):
        if args.policy == "random":
            action = rng.uniform(-1.0, 1.0, size=config.action_size)
        else:
            action = np.zeros(config.action_size)
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    dx = last_info["torso_x"] - start_x
    dy = last_info["torso_y"] - start_y
    drift_ratio = abs(dy) / max(abs(dx), 1e-6)
    reward_terms = last_info.get("reward_terms", {})
    print(
        "walk_smoke "
        f"obs_size={obs.shape[0]} action_size={config.action_size} "
        f"steps={last_info['step_count']} total_reward={total_reward:.3f} "
        f"x={last_info['torso_x']:.3f} y={last_info['torso_y']:.3f} "
        f"dx={dx:.3f} dy={dy:.3f} dy_per_dx={drift_ratio:.3f} "
        f"heading_sin={last_info['heading_sin']:.3f} heading_cos={last_info['heading_cos']:.3f} "
        f"heading_error={last_info['heading_error']:.3f} yaw_rate={last_info['yaw_rate']:.3f} "
        f"lateral_velocity={last_info['lateral_velocity']:.3f} "
        f"torso_height={last_info['torso_height']:.3f} "
        f"foot_contacts={last_info['foot_contact_count']} "
        f"contact_schedule_match={last_info.get('contact_schedule_match', 0.0):.3f} "
        f"reward_contact_schedule={reward_terms.get('contact_schedule', 0.0):.3f} "
        f"disk_contacts={last_info['disk_contact_count']} "
        f"forward_velocity={last_info['forward_velocity']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

