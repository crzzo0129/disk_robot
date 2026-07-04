def test_walk_reward_prefers_matching_forward_velocity():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig(command_velocity=0.45)
    good = compute_walk_reward(
        config=config,
        forward_velocity=0.45,
        lateral_velocity=0.0,
        vertical_velocity=0.0,
        angular_velocity_xy_mean_square=0.0,
        joint_velocity_mean_square=0.0,
        torso_height=0.32,
        upright=1.0,
        disk_contact_count=0,
        foot_contact_count=2,
        action_mean_square=0.01,
        action_delta_mean_square=0.01,
    )
    slow = compute_walk_reward(
        config=config,
        forward_velocity=0.0,
        lateral_velocity=0.0,
        vertical_velocity=0.0,
        angular_velocity_xy_mean_square=0.0,
        joint_velocity_mean_square=0.0,
        torso_height=0.32,
        upright=1.0,
        disk_contact_count=0,
        foot_contact_count=2,
        action_mean_square=0.01,
        action_delta_mean_square=0.01,
    )

    assert good.total > slow.total
    assert "velocity" in good.terms
    assert "forward" in good.terms


def test_walk_reward_penalizes_disk_ground_contact():
    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_reward import compute_walk_reward

    config = WalkTaskConfig()
    clean = compute_walk_reward(
        config=config,
        forward_velocity=config.command_velocity,
        lateral_velocity=0.0,
        vertical_velocity=0.0,
        angular_velocity_xy_mean_square=0.0,
        joint_velocity_mean_square=0.0,
        torso_height=0.32,
        upright=1.0,
        disk_contact_count=0,
        foot_contact_count=2,
        action_mean_square=0.0,
        action_delta_mean_square=0.0,
    )
    touching = compute_walk_reward(
        config=config,
        forward_velocity=config.command_velocity,
        lateral_velocity=0.0,
        vertical_velocity=0.0,
        angular_velocity_xy_mean_square=0.0,
        joint_velocity_mean_square=0.0,
        torso_height=0.32,
        upright=1.0,
        disk_contact_count=1,
        foot_contact_count=2,
        action_mean_square=0.0,
        action_delta_mean_square=0.0,
    )

    assert clean.total > touching.total
    assert touching.terms["disk_contact"] < 0.0
