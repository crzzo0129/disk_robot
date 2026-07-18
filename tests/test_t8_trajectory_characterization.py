import numpy as np


def _trajectory(lateral, yaw=0.0, failed=False, disk=False, force=1.0):
    steps = 4
    envs = len(lateral)
    x = np.linspace(0.0, 0.3, steps)[:, None].repeat(envs, axis=1)
    y = np.linspace(np.zeros(envs), np.asarray(lateral), steps)
    xy = np.stack((x, y), axis=-1)
    return {
        "xy": xy,
        "yaw": np.linspace(np.zeros(envs), np.full(envs, yaw), steps),
        "alive": np.ones((steps, envs), dtype=np.float32),
        "disk_contact": np.full((steps, envs), float(disk)),
        "failed": np.full((steps, envs), float(failed)),
        "action": np.full((steps, envs, 12), 0.2),
        "force": np.full((steps, envs, 12), force),
        "ctrl": np.zeros((steps, envs, 12)),
    }


def test_t8_trajectory_entry_imports_without_jax_and_defaults_to_long_paired_rollout():
    import sys

    jax_before = sys.modules.get("jax")
    brax_before = sys.modules.get("brax")
    from scripts import characterize_t8_trajectories

    assert sys.modules.get("jax") is jax_before
    assert sys.modules.get("brax") is brax_before
    args = characterize_t8_trajectories.parse_args(
        ["--teacher-run", "teacher", "--student-run", "student"]
    )
    assert args.steps == 1500
    assert args.seed_count == 4
    assert args.mujoco_gl == "disable"


def test_trajectory_summary_reports_signed_and_absolute_drift_and_limits():
    from scripts.characterize_t8_trajectories import summarize_trajectory

    report = summarize_trajectory(
        _trajectory([0.03, -0.01], yaw=0.2, force=2.98),
        dt=0.02,
        torque_limit=3.0,
        ctrl_low=-np.ones(12),
        ctrl_high=np.ones(12),
    )

    np.testing.assert_allclose(report["mean_lateral_displacement_m"], 0.01)
    np.testing.assert_allclose(report["mean_absolute_lateral_displacement_m"], 0.02)
    np.testing.assert_allclose(report["maximum_absolute_lateral_displacement_m"], 0.03)
    assert report["mean_absolute_yaw_change_rad"] == 0.2
    assert report["force_saturation_fraction"] == 1.0
    assert len(report["per_environment"]) == 2


def test_characterization_gate_routes_student_only_regression_to_diagnosis():
    from scripts.characterize_t8_trajectories import characterize_gate, summarize_trajectory

    def summary(lateral):
        return summarize_trajectory(
            _trajectory([lateral, lateral]),
            dt=0.02,
            torque_limit=3.0,
            ctrl_low=-np.ones(12),
            ctrl_high=np.ones(12),
        )

    gate = characterize_gate(
        {"ik": summary(0.01), "teacher": summary(0.012), "student": summary(0.05)}
    )

    assert gate["student_materially_worse_than_teacher"]
    assert gate["recommendation"] == "diagnose_t8_retention_before_t9"


def test_environment_has_opt_in_fixed_phase_without_changing_training_default():
    from disk_robot.teacher_student_config import ForwardTeacherStudentConfig

    assert ForwardTeacherStudentConfig().fixed_reset_phase is None
    source = open("disk_robot_mjx/teacher_student_env.py", encoding="utf-8").read()
    assert "self.config.fixed_reset_phase is not None" in source
