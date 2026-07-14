def _reward_inputs(**overrides):
    from disk_robot.walk_reward import WalkRewardInputs

    values = dict(
        velocity_x=0.2,
        velocity_y=0.0,
        yaw_rate=0.0,
        command_x=0.2,
        command_y=0.0,
        command_yaw=0.0,
        vertical_velocity=0.0,
        roll_pitch_rate_mean_square=0.0,
        joint_velocity_mean_square=0.0,
        upright=1.0,
        disk_contact_count=0,
        action_mean_square=0.0,
        action_delta_mean_square=0.0,
    )
    values.update(overrides)
    return WalkRewardInputs(**values)


def test_nonzero_command_strongly_prefers_matching_velocity_over_standing():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig()
    matching = compute_walk_reward(config=config, inputs=_reward_inputs())
    standing = compute_walk_reward(config=config, inputs=_reward_inputs(velocity_x=0.0))

    assert matching.total > standing.total + 0.8
    assert matching.terms["velocity_xy"] > standing.terms["velocity_xy"]
    assert matching.terms["directional_progress"] > standing.terms["directional_progress"]


def test_reward_tracks_lateral_and_yaw_commands_instead_of_fixed_heading():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig()
    matching = compute_walk_reward(
        config=config,
        inputs=_reward_inputs(velocity_y=-0.15, yaw_rate=0.7, command_y=-0.15, command_yaw=0.7),
    )
    wrong = compute_walk_reward(
        config=config,
        inputs=_reward_inputs(velocity_y=0.15, yaw_rate=-0.7, command_y=-0.15, command_yaw=0.7),
    )

    assert matching.total > wrong.total
    assert "heading" not in matching.terms


def test_zero_command_stand_bonus_only_applies_at_zero_command():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig()
    zero = compute_walk_reward(
        config=config,
        inputs=_reward_inputs(velocity_x=0.0, command_x=0.0),
    )
    moving_command = compute_walk_reward(
        config=config,
        inputs=_reward_inputs(velocity_x=0.0, command_x=0.1),
    )

    assert zero.terms["stand"] > 0.0
    assert moving_command.terms["stand"] == 0.0


def test_disk_contact_action_rate_and_failure_are_penalties():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig()
    clean = compute_walk_reward(config=config, inputs=_reward_inputs())
    unsafe = compute_walk_reward(
        config=config,
        inputs=_reward_inputs(disk_contact_count=1, action_delta_mean_square=1.0, failed=True),
    )

    assert clean.total > unsafe.total
    assert unsafe.terms["disk_contact"] < 0.0
    assert unsafe.terms["action_delta"] < 0.0
    assert unsafe.terms["termination"] < 0.0


def test_teacher_imitation_reward_prefers_teacher_action():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig(reward_teacher_imitation=1.0)
    matching = compute_walk_reward(config=config, inputs=_reward_inputs(teacher_action_error=0.0))
    random = compute_walk_reward(config=config, inputs=_reward_inputs(teacher_action_error=1.0))

    assert matching.terms["teacher_imitation"] > random.terms["teacher_imitation"]
    assert matching.total > random.total
