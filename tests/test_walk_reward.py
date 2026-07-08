def _reward_inputs(**overrides):
    from disk_robot.walk_reward import WalkRewardInputs

    values = dict(
        forward_velocity=0.1,
        lateral_velocity=0.0,
        yaw_rate=0.0,
        heading_error=0.0,
        vertical_velocity=0.0,
        roll_pitch_rate_mean_square=0.0,
        joint_velocity_mean_square=0.0,
        torso_height=0.406,
        upright=1.0,
        disk_contact_count=0,
        foot_contact_count=2,
        contact_schedule_match=1.0,
        action_mean_square=0.0,
        action_delta_mean_square=0.0,
    )
    values.update(overrides)
    return WalkRewardInputs(**values)


def test_walk_reward_prefers_matching_forward_velocity():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig(command_velocity=0.45)
    good = compute_walk_reward(config=config, inputs=_reward_inputs(forward_velocity=0.45, action_mean_square=0.01, action_delta_mean_square=0.01))
    slow = compute_walk_reward(config=config, inputs=_reward_inputs(forward_velocity=0.0, action_mean_square=0.01, action_delta_mean_square=0.01))

    assert good.total > slow.total
    assert "velocity" in good.terms
    assert "forward" in good.terms


def test_walk_reward_penalizes_disk_ground_contact():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig()
    clean = compute_walk_reward(config=config, inputs=_reward_inputs(forward_velocity=config.command_velocity))
    touching = compute_walk_reward(
        config=config,
        inputs=_reward_inputs(forward_velocity=config.command_velocity, disk_contact_count=1),
    )

    assert clean.total > touching.total
    assert touching.terms["disk_contact"] < 0.0


def test_walk_reward_rewards_contact_schedule_match():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig(reward_contact_schedule=0.2)
    matching = compute_walk_reward(config=config, inputs=_reward_inputs(forward_velocity=config.command_velocity))
    mismatching = compute_walk_reward(
        config=config,
        inputs=_reward_inputs(forward_velocity=config.command_velocity, contact_schedule_match=0.0),
    )

    assert matching.total > mismatching.total
    assert matching.terms["contact_schedule"] > mismatching.terms["contact_schedule"]


def test_walk_reward_penalizes_foot_slip():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig(penalty_foot_slip=0.1)
    clean = compute_walk_reward(config=config, inputs=_reward_inputs(forward_velocity=config.command_velocity))
    slipping = compute_walk_reward(
        config=config,
        inputs=_reward_inputs(forward_velocity=config.command_velocity, foot_slip_mean_square=1.0),
    )

    assert clean.total > slipping.total
    assert slipping.terms["foot_slip"] < 0.0


def test_timeout_is_not_a_failure_penalty():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig(penalty_termination=1000.0)
    survived = compute_walk_reward(config=config, inputs=_reward_inputs(failed=False))
    failed = compute_walk_reward(config=config, inputs=_reward_inputs(failed=True))

    assert survived.terms["termination"] == 0.0
    assert failed.terms["termination"] == -1000.0
    assert survived.total > failed.total
