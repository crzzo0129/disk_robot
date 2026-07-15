from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from disk_robot.student_policy import load_student_policy, make_student_policy_jax
from disk_robot.teacher_student_config import ForwardTeacherStudentConfig
from disk_robot_mjx.pipeline import configure_cloud_runtime
from disk_robot_mjx.teacher_student_env import DEFAULT_XML, make_forward_teacher_student_env


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a distilled forward student with IK disabled.")
    parser.add_argument("policy", type=Path)
    parser.add_argument("--xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=90000)
    parser.add_argument("--mujoco-gl", default="egl")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_cloud_runtime(mujoco_gl=args.mujoco_gl, verbose=True)
    try:
        import jax
    except ImportError as exc:
        raise SystemExit("Evaluation requires the mjx312 JAX/MJX environment") from exc

    artifact = load_student_policy(args.policy)
    metadata = artifact.metadata
    config = ForwardTeacherStudentConfig(**metadata.get("config", {}))
    env = make_forward_teacher_student_env(
        "student", config=config, xml_path=args.xml_path, seed=args.seed
    )
    policy = jax.jit(make_student_policy_jax(artifact))
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step))
    state = reset_batch(jax.random.split(jax.random.PRNGKey(args.seed), args.envs))

    def eval_step(current_state, _):
        next_state = step_batch(current_state, policy(current_state.obs))
        alive = 1.0 - current_state.done
        return next_state, (
            next_state.metrics["velocity_x"],
            next_state.metrics["velocity_error"],
            next_state.metrics["roll_pitch_rate_rms"],
            next_state.metrics["failed"],
            alive,
        )

    _, values = jax.lax.scan(eval_step, state, (), length=args.steps)
    vx, velocity_error, roll_pitch_rate, failed, alive = [
        np.asarray(jax.device_get(value)) for value in values
    ]
    denominator = max(float(np.sum(alive)), 1.0)
    report = {
        "mean_velocity_x": float(np.sum(vx * alive) / denominator),
        "mean_velocity_error": float(np.sum(velocity_error * alive) / denominator),
        "mean_roll_pitch_rate_rms": float(np.sum(roll_pitch_rate * alive) / denominator),
        "failure_rate": float(np.mean(np.max(failed, axis=0))),
        "mean_alive_steps": float(np.mean(np.sum(alive, axis=0))),
        "ik_runtime_enabled": False,
        "teacher_observation_enabled": False,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
