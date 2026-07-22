import json

import numpy as np
import pytest


def test_t9_teacher_entry_has_explicit_grid_and_never_distills_inline():
    from scripts import train_t9_forward_teacher

    args, passthrough = train_t9_forward_teacher.parse_args(["--smoke"])

    assert args.speed_anchors == [0.0, 0.04, 0.06, 0.08, 0.10]
    assert args.teacher_steps == 1_500_000
    assert args.mujoco_gl == "disable"
    assert args.smoke
    assert passthrough == []
    source = open("scripts/train_t9_forward_teacher.py", encoding="utf-8").read()
    assert '"--teacher-only"' in source
    assert '"--teacher-disturbances"' in source
    assert '"--command-vx-grid"' in source


def test_generic_pipeline_rejects_grid_distillation_before_loading_jax():
    from scripts.train_forward_teacher_student import main

    with pytest.raises(SystemExit, match="requires --teacher-only"):
        main(["--command-vx-grid", "0", "0.04"])


def test_t9_preliminary_teacher_is_not_silently_accepted(tmp_path):
    from scripts.train_t9_forward_teacher import _mark_grid_validation_pending

    teacher = tmp_path / "teacher"
    teacher.mkdir()
    evaluation = teacher / "evaluation.json"
    evaluation.write_text(json.dumps({"accepted": True}), encoding="utf-8")

    _mark_grid_validation_pending(tmp_path)
    report = json.loads(evaluation.read_text(encoding="utf-8"))

    assert report["preliminary_aggregate_accepted"]
    assert not report["accepted"]
    assert report["grid_validation_pending"]


def test_t9_speed_gate_requires_nominal_disturbed_and_long_horizon():
    from scripts.evaluate_t9_teacher_grid import _speed_gate

    baseline = {
        "failure_rate": 0.0,
        "mean_velocity_error": 0.01,
        "mean_roll_pitch_rate_rms": 0.2,
        "mean_abs_velocity_y": 0.02,
        "mean_abs_yaw_rate": 0.1,
        "mean_post_push_velocity_error": 0.08,
        "mean_recovery_time": 0.5,
        "mean_disk_contacts": 0.0,
        "mean_forward_distance": 0.8,
        "reward_per_step": 1.5,
    }
    ppo = dict(baseline)
    long_baseline = {
        "failure_rate": 0.0,
        "disk_contact_environment_rate": 0.0,
        "force_saturation_fraction": 0.0,
        "mean_absolute_lateral_displacement_m": 0.2,
        "mean_absolute_yaw_change_rad": 0.2,
    }
    long_ppo = dict(long_baseline)

    passing = _speed_gate(
        0.08, ppo, baseline, ppo, baseline, long_ppo, long_baseline
    )
    assert passing["accepted"]

    long_ppo["mean_absolute_lateral_displacement_m"] = 0.46
    failing = _speed_gate(
        0.08, ppo, baseline, ppo, baseline, long_ppo, long_baseline
    )
    assert not failing["accepted"]
    assert not failing["long_horizon_safe"]


def test_t9_reference_bank_builds_stop_and_moving_tables():
    from disk_robot.t9_command import build_t9_reference_bank
    from disk_robot_mjx.teacher_student_env import DEFAULT_XML

    bank = build_t9_reference_bank(
        DEFAULT_XML, (0.0, 0.04, 0.10), samples=16
    )

    assert bank.joint_targets.shape == (3, 16, 12)
    assert bank.desired_contacts.shape == (3, 16, 4)
    assert np.max(np.abs(bank.joint_targets[0] - bank.stand_q)) < 1e-5
    assert np.max(np.abs(bank.joint_targets[-1] - bank.stand_q)) > 0.01


def test_t9_student_entry_is_bc_only_and_uses_138d_contract():
    import sys

    jax_before = sys.modules.get("jax")
    brax_before = sys.modules.get("brax")
    from scripts import distill_t9_forward_student

    args = distill_t9_forward_student.parse_args(
        ["--teacher-run", "teacher", "--t8-run", "t8"]
    )
    assert sys.modules.get("jax") is jax_before
    assert sys.modules.get("brax") is brax_before
    assert args.dataset_samples == 196_608
    source = open("scripts/distill_t9_forward_student.py", encoding="utf-8").read()
    assert "ppo.train(" not in source
    assert "_collect_teacher_dataset" in source
    assert '"student_policy_obs"' in source
    assert '"T9_FORWARD_COMMAND_BC"' in source
    assert "student_policy_t9_forward_command_bc.npz" in source


def test_t9_student_speed_gate_includes_disturbance_and_long_retention():
    from scripts.distill_t9_forward_student import _student_speed_gate

    teacher = {
        "failure_rate": 0.0,
        "mean_velocity_error": 0.01,
        "mean_roll_pitch_rate_rms": 0.2,
        "mean_abs_velocity_y": 0.02,
        "mean_abs_yaw_rate": 0.1,
        "mean_post_push_velocity_error": 0.05,
        "mean_recovery_time": 0.4,
        "mean_disk_contacts": 0.0,
    }
    student = dict(teacher)
    long_teacher = {
        "failure_rate": 0.0,
        "disk_contact_environment_rate": 0.0,
        "force_saturation_fraction": 0.0,
        "mean_absolute_lateral_displacement_m": 0.2,
        "mean_absolute_yaw_change_rad": 0.2,
    }
    long_student = dict(long_teacher)

    gate = _student_speed_gate(
        0.08,
        student,
        teacher,
        student,
        teacher,
        long_student,
        long_teacher,
    )
    assert gate["accepted"]
    student["mean_recovery_time"] = 1.0
    gate = _student_speed_gate(
        0.08,
        student,
        teacher,
        student,
        teacher,
        long_student,
        long_teacher,
    )
    assert not gate["accepted"]
    assert not gate["disturbed_retention"]


def test_t9_environment_owns_episode_command_and_reference_bank():
    source = open("disk_robot_mjx/teacher_student_env.py", encoding="utf-8").read()

    assert '"command": command' in source
    assert source.count('"command_vx"') >= 2
    assert 'state.info["command"]' in source
    assert "interpolate_reference_bank_jax" in source
    assert "command_active.astype" in source
    assert "student_current_command_only" in source


def test_t9_student_retention_uses_paired_phase_zero_seeds():
    source = open("scripts/distill_t9_forward_student.py", encoding="utf-8").read()

    assert "fixed_reset_phase=0.0" in source
    assert "paired_teacher_env" in source
    assert "paired_teacher_disturbed" in source
    assert "retention_seed" in source
    assert '"student", config=t8_config, xml_path=xml_path, seed=0' in source
    assert '"student", config=t9_retention_config, xml_path=xml_path, seed=0' in source
