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
    assert config.privileged_size == 35
    assert config.teacher_observation_size == 227
    assert len(config.student_action_scale) == 12
    assert len(config.residual_scale) == 12
