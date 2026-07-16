import numpy as np


def test_phase_clock_uses_only_command_and_controller_time():
    from disk_robot.phase_clock import (
        PhaseClockConfig,
        PhaseClockState,
        phase_clock_observation,
        update_phase_clock,
    )

    config = PhaseClockConfig(
        frequency_hz=1.2,
        startup_blend_steps=25,
        command_deadzone=0.01,
    )
    state = update_phase_clock(
        PhaseClockState(),
        command=(0.08, 0.0, 0.0),
        dt=0.02,
        config=config,
    )

    np.testing.assert_allclose(state.phase, 0.024, atol=1e-9)
    assert state.gait_blend == 0.04
    observation = phase_clock_observation(state)
    assert observation.shape == (3,)
    assert np.all(np.isfinite(observation))


def test_phase_clock_freezes_and_blends_out_in_the_command_deadzone():
    from disk_robot.phase_clock import PhaseClockConfig, PhaseClockState, update_phase_clock

    config = PhaseClockConfig(frequency_hz=1.2, startup_blend_steps=10)
    state = PhaseClockState(phase=0.3, gait_blend=0.5)
    next_state = update_phase_clock(
        state,
        command=(0.0, 0.0, 0.0),
        dt=0.02,
        config=config,
    )

    assert next_state.phase == 0.3
    assert next_state.gait_blend == 0.4


def test_phase_student_entry_enables_internal_clock_without_loading_jax():
    import sys

    jax_before = sys.modules.get("jax")
    brax_before = sys.modules.get("brax")
    from scripts import distill_phase_student

    assert sys.modules.get("jax") is jax_before
    assert sys.modules.get("brax") is brax_before
    source = open("scripts/distill_phase_student.py", encoding="utf-8").read()
    assert '"--phase-conditioned"' in source


def test_phase_student_keeps_teacher_observation_contract_and_has_no_contact_input():
    config_source = open("disk_robot/teacher_student_config.py", encoding="utf-8").read()
    env_source = open("disk_robot_mjx/teacher_student_env.py", encoding="utf-8").read()
    distill_source = open("scripts/distill_forward_student.py", encoding="utf-8").read()

    assert "student_policy_observation_size" in config_source
    assert 'student_observation_key = (' in distill_source
    assert '"student_policy_obs"' in distill_source
    assert "requires_foot_contact" in distill_source
    assert "self._student_policy_obs(" in env_source
    assert "student_history, internal_state" in env_source


def test_phase_diagnosis_compares_teacher_oracle_and_learned_student():
    source = open("scripts/diagnose_phase_student.py", encoding="utf-8").read()
    pipeline_source = open(
        "scripts/train_forward_teacher_student.py", encoding="utf-8"
    ).read()

    assert "_evaluate_teacher(" in source
    assert "_evaluate_oracle_student(" in source
    assert "_evaluate_student(" in source
    assert "oracle_preserves_teacher_velocity" in source
    assert "def _evaluate_oracle_student" in pipeline_source
    assert "teacher_action_to_student_action" in pipeline_source


def test_failure_audit_imports_without_jax_and_exposes_all_four_diagnostics():
    import sys

    jax_before = sys.modules.get("jax")
    brax_before = sys.modules.get("brax")
    from scripts import audit_phase_student_failure

    assert sys.modules.get("jax") is jax_before
    assert sys.modules.get("brax") is brax_before
    args = audit_phase_student_failure.parse_args(
        ["--teacher-run", "teacher", "--student-run", "student", "--smoke"]
    )
    assert args.smoke
    assert args.noise_levels == [0.0, 0.001, 0.002, 0.005, 0.01]
    source = open(
        "scripts/audit_phase_student_failure.py", encoding="utf-8"
    ).read()
    assert "def _offline_error_audit" in source
    assert "def _evaluate_noisy_oracle" in source
    assert "def _paired_divergence_rollout" in source
    assert "def _nearest_neighbor_audit" in source
    assert '"stage": "T7_FAILURE_AUDIT"' in source


def test_failure_audit_splits_offline_error_by_joint_and_phase():
    from disk_robot.student_policy import StudentPolicyArtifact
    from scripts.audit_phase_student_failure import _offline_error_audit

    observations = np.zeros((4, 195), dtype=np.float32)
    phases = np.asarray((0.1, 0.2, 0.6, 0.7), dtype=np.float32)
    observations[:, -3] = np.sin(2.0 * np.pi * phases)
    observations[:, -2] = np.cos(2.0 * np.pi * phases)
    observations[:, -1] = 1.0
    labels = np.zeros((4, 12), dtype=np.float32)
    labels[:, 0] = 0.02
    artifact = StudentPolicyArtifact(
        params=((np.zeros((195, 12), dtype=np.float32), np.zeros(12, dtype=np.float32)),),
        obs_mean=np.zeros(195, dtype=np.float32),
        obs_std=np.ones(195, dtype=np.float32),
        metadata={},
    )

    report = _offline_error_audit(
        artifact,
        observations,
        labels,
        np.ones(12, dtype=np.float32),
        phase_bins=2,
    )

    assert report["samples"] == 4
    assert report["largest_rmse_joint"] == "leg_front_r_1"
    assert report["largest_bias_joint"] == "leg_front_r_1"
    assert [entry["samples"] for entry in report["phase_bins"]] == [2, 2]
    np.testing.assert_allclose(
        report["per_joint"][0]["action_error"]["mean"], -0.02, atol=1e-7
    )


def test_failure_audit_nearest_neighbor_uses_cross_environment_pairs():
    from scripts.audit_phase_student_failure import _nearest_neighbor_audit

    observations = np.asarray(
        [[0.0, 0.0], [0.01, 0.0], [0.0, 0.0], [0.01, 0.0]], dtype=np.float32
    )
    teacher_labels = np.asarray(
        [[0.0], [0.0], [0.1], [0.1]], dtype=np.float32
    )
    student_actions = np.zeros((4, 1), dtype=np.float32)
    report = _nearest_neighbor_audit(
        observations,
        teacher_labels,
        student_actions,
        np.asarray([0, 0, 1, 1]),
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
        sample_count=4,
        seed=0,
    )

    assert report["cross_environment_only"]
    assert report["samples"] == 4
    np.testing.assert_allclose(
        report["teacher_label_disagreement_rms"]["mean"], 0.1, atol=1e-7
    )
