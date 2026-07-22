import numpy as np


def test_ik_reference_uses_xml_stand_and_is_periodic():
    import mujoco

    from disk_robot.ik_reference import IKReferenceSpec, build_ik_reference
    from disk_robot.model_contract import resolve_model_contract
    from disk_robot.walk_env import DEFAULT_XML

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    xml_stand = resolve_model_contract(model).stand_q
    reference = build_ik_reference(DEFAULT_XML, IKReferenceSpec(samples=32))

    assert reference.joint_targets.shape == (32, 12)
    assert reference.desired_contacts.shape == (32, 4)
    assert np.allclose(reference.stand_q, xml_stand)
    assert np.max(np.abs(reference.joint_targets)) > 0.01
    assert np.max(np.abs(reference.joint_targets[0] - reference.joint_targets[-1])) < 0.1


def test_teacher_student_observation_contract_is_explicit():
    from disk_robot.teacher_student_config import ForwardTeacherStudentConfig

    config = ForwardTeacherStudentConfig()

    assert config.student_frame_size == 48
    assert config.student_observation_size == 192
    assert config.student_policy_observation_size == 192
    assert config.privileged_size == 39
    assert config.teacher_observation_size == 231
    assert config.recovery_velocity_ema_alpha == 0.10
    assert config.recovery_required_steps == 4
    assert len(config.student_action_scale) == 12
    assert len(config.residual_scale) == 12

    phase_config = ForwardTeacherStudentConfig(student_phase_conditioned=True)
    assert phase_config.student_internal_state_size == 3
    assert phase_config.student_policy_observation_size == 195
    assert phase_config.teacher_observation_size == 231

    no_action_history_config = ForwardTeacherStudentConfig(
        student_phase_conditioned=True,
        student_previous_action_input=False,
    )
    assert no_action_history_config.student_policy_frame_size == 36
    assert no_action_history_config.student_policy_sensor_history_size == 144
    assert no_action_history_config.student_policy_observation_size == 147
    assert no_action_history_config.student_observation_size == 192
    assert no_action_history_config.teacher_observation_size == 231


def test_reference_can_be_built_from_an_in_memory_structure_variant():
    import mujoco

    from disk_robot.ik_reference import IKReferenceSpec, build_ik_reference_from_model
    from disk_robot.structure_variants import StructureVariant, apply_structure_variant
    from disk_robot.walk_env import DEFAULT_XML

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
    apply_structure_variant(model, StructureVariant(hip_y=0.085, leg_scale=0.85))
    reference = build_ik_reference_from_model(model, IKReferenceSpec(samples=16))

    assert reference.joint_targets.shape == (16, 12)
    assert np.all(np.isfinite(reference.joint_targets))


def test_t9_contract_keeps_four_physical_frames_and_one_current_command():
    from disk_robot.t9_command import make_t9_config

    config = make_t9_config()

    assert config.command_vx_values == (0.0, 0.04, 0.06, 0.08, 0.10)
    assert config.student_policy_frame_size == 33
    assert config.student_policy_sensor_history_size == 132
    assert config.student_current_command_size == 3
    assert config.student_internal_state_size == 3
    assert config.student_policy_observation_size == 138
    assert config.teacher_observation_size == 231


def test_t9_reference_specs_are_genuinely_speed_conditioned():
    from disk_robot.t9_command import forward_reference_specs

    specs = forward_reference_specs(samples=32)

    assert [spec.stride_length for spec in specs] == sorted(
        spec.stride_length for spec in specs
    )
    assert specs[0].stride_length == 0.0
    assert specs[0].step_height == 0.0
    assert specs[-1].stride_length > specs[1].stride_length
    assert {spec.frequency for spec in specs} == {1.2}


def test_reference_bank_interpolates_speed_and_phase_independently():
    from disk_robot.ik_reference import interpolate_reference_bank_jax

    commands = np.asarray([0.0, 0.1], dtype=np.float32)
    tables = np.asarray(
        [
            [[0.0], [2.0]],
            [[10.0], [12.0]],
        ],
        dtype=np.float32,
    )

    value = interpolate_reference_bank_jax(
        np, commands, tables, command_vx=0.05, phase=0.25
    )

    np.testing.assert_allclose(value, [6.0])
