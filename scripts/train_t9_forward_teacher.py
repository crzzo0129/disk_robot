from __future__ import annotations

import argparse
import json
from pathlib import Path

from disk_robot.t9_command import T9_FORWARD_SPEED_ANCHORS
from scripts.train_forward_teacher_student import main as train_teacher_student


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="T9 stage 1: train an episode-fixed vx-conditioned robust PPO Teacher."
    )
    parser.add_argument("--out", type=Path, default=Path("mjx_runs") / "teacher_t9_vx_grid_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--speed-anchors",
        type=float,
        nargs="+",
        default=list(T9_FORWARD_SPEED_ANCHORS),
    )
    parser.add_argument("--teacher-steps", type=int, default=1_500_000)
    parser.add_argument("--teacher-envs", type=int, default=2048)
    parser.add_argument("--teacher-eval-envs", type=int, default=256)
    parser.add_argument("--teacher-restore", type=Path, default=None)
    parser.add_argument("--xml-path", type=Path, default=None)
    parser.add_argument("--mujoco-gl", default="disable")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-xla-triton", dest="xla_triton", action="store_false")
    parser.set_defaults(xla_triton=True)
    return parser.parse_known_args(argv)


def _mark_grid_validation_pending(out: Path):
    evaluation_path = out.expanduser().resolve() / "teacher" / "evaluation.json"
    if not evaluation_path.exists():
        return
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    report["stage"] = "T9_FORWARD_COMMAND_TEACHER_PRELIMINARY"
    report["preliminary_aggregate_accepted"] = bool(report.get("accepted", False))
    report["accepted"] = False
    report["grid_validation_pending"] = True
    report["rejection_reason"] = "t9_speed_grid_evaluation_pending"
    evaluation_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "stage=t9_teacher_preliminary "
        f"aggregate_accepted={report['preliminary_aggregate_accepted']} "
        f"grid_validation_pending=True report={evaluation_path}",
        flush=True,
    )


def main(argv=None):
    (args, passthrough) = parse_args(argv)
    forwarded = [
        "--out",
        str(args.out),
        "--seed",
        str(args.seed),
        "--teacher-steps",
        str(args.teacher_steps),
        "--teacher-envs",
        str(args.teacher_envs),
        "--teacher-eval-envs",
        str(args.teacher_eval_envs),
        "--teacher-minibatches",
        "8",
        "--teacher-selection-mode",
        "robust",
        "--teacher-disturbances",
        "--teacher-only",
        "--command-vx-grid",
        *(str(value) for value in args.speed_anchors),
        "--mujoco-gl",
        args.mujoco_gl,
    ]
    if args.teacher_restore is not None:
        forwarded.extend(("--teacher-restore", str(args.teacher_restore)))
    if args.xml_path is not None:
        forwarded.extend(("--xml-path", str(args.xml_path)))
    if args.smoke:
        forwarded.append("--smoke")
    if not args.xla_triton:
        forwarded.append("--no-xla-triton")
    train_teacher_student([*forwarded, *passthrough])
    _mark_grid_validation_pending(args.out)


if __name__ == "__main__":
    main()
