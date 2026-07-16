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
