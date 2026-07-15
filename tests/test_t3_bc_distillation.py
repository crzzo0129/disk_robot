import json
from types import SimpleNamespace


def test_t3_entry_imports_without_jax_or_brax():
    import sys

    jax_before = sys.modules.get("jax")
    brax_before = sys.modules.get("brax")
    from scripts import distill_forward_student

    assert sys.modules.get("jax") is jax_before
    assert sys.modules.get("brax") is brax_before
    args = distill_forward_student.parse_args(["--teacher-run", "teacher"])
    assert args.dataset_samples == 131_072
    assert args.nominal_fraction == 0.5
    assert args.student_updates == 20_000


def test_t3_reconstructs_the_accepted_teacher_environment():
    from scripts.distill_forward_student import _config_from_teacher_run

    config = _config_from_teacher_run(
        {
            "episode_length": 400,
            "command_vx": 0.08,
            "kp": 11.0,
            "kd": 0.5,
            "torque_limit": 2.5,
            "startup_steps": 30,
            "residual_scale_multiplier": 0.25,
            "residual_filter_alpha": 0.2,
            "penalty_residual": 0.1,
            "penalty_residual_rate": 0.04,
            "teacher_disturbances": True,
            "push_velocity_x": 0.6,
            "push_velocity_y": 0.45,
            "recovery_required_steps": 5,
        }
    )

    assert config.max_episode_steps == 400
    assert config.actuator_kp == 11.0
    assert config.disturbance_enabled
    assert config.push_velocity_x == 0.6
    assert config.recovery_required_steps == 5
    assert config.residual_scale[0] == 0.025


def test_t3_refuses_unaccepted_or_non_ppo_teacher(tmp_path):
    import pytest

    from scripts.distill_forward_student import _load_accepted_teacher_run

    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    (tmp_path / "run_config.json").write_text("{}", encoding="utf-8")
    (teacher_dir / "selection.json").write_text("{}", encoding="utf-8")
    (teacher_dir / "evaluation.json").write_text(
        json.dumps({"accepted": False, "selected_source": "ppo"}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="accepted=true"):
        _load_accepted_teacher_run(tmp_path)

    (teacher_dir / "evaluation.json").write_text(
        json.dumps({"accepted": True, "selected_source": "ik_baseline"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="accepted PPO Teacher"):
        _load_accepted_teacher_run(tmp_path)


def test_t3_bc_gate_requires_nominal_and_disturbed_retention():
    from scripts.distill_forward_student import _bc_acceptance

    teacher_nominal = {
        "mean_velocity_x": 0.08,
        "failure_rate": 0.0,
        "mean_roll_pitch_rate_rms": 0.20,
        "mean_abs_velocity_y": 0.03,
        "mean_abs_yaw_rate": 0.15,
    }
    teacher_disturbed = {
        "failure_rate": 0.0,
        "mean_post_push_velocity_error": 0.08,
        "mean_recovery_time": 0.5,
        "mean_forward_distance": 0.75,
        "mean_disk_contacts": 0.0,
    }
    args = SimpleNamespace(
        nominal_vx_tolerance=0.015,
        nominal_failure_tolerance=0.05,
        nominal_roll_pitch_tolerance=0.10,
        nominal_lateral_tolerance=0.015,
        nominal_yaw_tolerance=0.05,
        disturbed_failure_tolerance=0.05,
        disturbed_post_error_tolerance=0.03,
        disturbed_recovery_tolerance=0.5,
        disturbed_distance_tolerance=0.10,
        disturbed_disk_tolerance=0.02,
    )
    nominal = dict(teacher_nominal)
    disturbed = dict(teacher_disturbed)

    passing = _bc_acceptance(
        nominal,
        disturbed,
        {
            "nominal_evaluation": teacher_nominal,
            "disturbed_evaluation": teacher_disturbed,
        },
        args,
    )
    assert passing["accepted"]

    disturbed["mean_recovery_time"] = 1.01
    failing = _bc_acceptance(
        nominal,
        disturbed,
        {
            "nominal_evaluation": teacher_nominal,
            "disturbed_evaluation": teacher_disturbed,
        },
        args,
    )
    assert not failing["accepted"]
    assert failing["nominal_preserved"]
    assert not failing["disturbed_preserved"]


def test_t3_is_bc_only_and_student_eval_reports_recovery_metrics():
    t3_source = open("scripts/distill_forward_student.py", encoding="utf-8").read()
    pipeline_source = open(
        "scripts/train_forward_teacher_student.py", encoding="utf-8"
    ).read()

    assert "_collect_dagger_dataset" not in t3_source
    assert "ppo.train(" not in t3_source
    assert "from brax.training.acme import running_statistics" in t3_source
    assert '"stage": "T3_BC"' in t3_source
    assert '"mean_post_push_velocity_error"' in pipeline_source
    assert '"mean_recovery_time"' in pipeline_source
    assert '"push_coverage"' in pipeline_source
