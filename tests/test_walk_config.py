def test_walk_config_declares_pupper_action_and_hardware_observation():
    from disk_robot.model_contract import FOOT_BODY_NAMES, JOINT_NAMES
    from disk_robot.walk_config import WalkTaskConfig

    config = WalkTaskConfig()

    assert len(JOINT_NAMES) == 12
    assert len(FOOT_BODY_NAMES) == 4
    assert config.action_size == 12
    assert config.observation_frame_size == 48
    assert config.observation_history == 4
    assert config.observation_size == 192
    assert not hasattr(config, "use_open_loop_gait")
    assert config.teacher_blend == 0.0
    assert config.reward_teacher_imitation == 0.0


def test_command_profiles_expand_task_space_without_changing_action_scale():
    from disk_robot.walk_config import command_profile

    forward = command_profile("forward")
    omni = command_profile("omni")
    full = command_profile("full")

    assert forward.command_vy_min == forward.command_vy_max == 0.0
    assert forward.command_zero_probability == 0.0
    assert forward.command_vx_min >= 0.15
    assert omni.command_vy_min < 0.0 < omni.command_vy_max
    assert full.command_vx_max > omni.command_vx_max
    assert forward.action_scale == omni.action_scale == full.action_scale
