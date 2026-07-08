def test_walk_config_declares_12_dof_action_and_named_feet():
    from disk_robot.walk_config import FOOT_GEOMS, JOINT_NAMES, WalkTaskConfig

    config = WalkTaskConfig()

    assert len(JOINT_NAMES) == 12
    assert len(FOOT_GEOMS) == 4
    assert config.action_size == 12
    assert config.command_velocity > 0.0


def test_walk_config_observation_size_matches_declared_layout():
    from disk_robot.walk_config import WalkTaskConfig

    config = WalkTaskConfig()

    assert config.observation_frame_size == 60
    assert config.observation_size == 60 * config.observation_history
