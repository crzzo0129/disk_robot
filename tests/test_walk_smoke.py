def test_walk_env_imports_without_loading_mujoco():
    import sys

    mujoco_before = sys.modules.get("mujoco")
    from disk_robot.walk_env import DEFAULT_XML, DiskRobotWalkEnv

    assert sys.modules.get("mujoco") is mujoco_before
    assert DEFAULT_XML.name == "pupper_v3_disk_structure_candidate.xml"
    assert DiskRobotWalkEnv.__name__ == "DiskRobotWalkEnv"


def test_target_xml_contract_and_static_pose_residual_action():
    import numpy as np
    import pytest

    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_env import DiskRobotWalkEnv

    config = WalkTaskConfig(action_repeat=1, command_zero_probability=1.0)
    env = DiskRobotWalkEnv(config=config, seed=1)
    obs, info = env.reset()

    assert obs.shape == (192,)
    assert np.allclose(info["target_ctrl"], env.contract.stand_q)
    foot_bottom = (
        env.data.geom_xpos[env.contract.foot_geom_ids, 2] - env.contract.foot_radii
    )
    assert np.min(foot_bottom) == pytest.approx(config.reset_foot_clearance, abs=1e-6)
    action = np.ones(config.action_size)
    _, reward, terminated, truncated, info = env.step(action)
    expected = np.clip(
        env.contract.stand_q + np.asarray(config.action_scale) * action,
        env.contract.ctrl_low,
        env.contract.ctrl_high,
    )
    assert np.allclose(info["target_ctrl"], expected)
    assert isinstance(float(reward), float)
    assert terminated in (False, True)
    assert truncated in (False, True)


def test_command_is_in_each_observation_frame_and_resamples():
    import numpy as np

    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_env import DiskRobotWalkEnv

    config = WalkTaskConfig(command_resample_steps=1, command_zero_probability=0.0, action_repeat=1)
    env = DiskRobotWalkEnv(config=config, seed=4)
    obs, _ = env.reset()
    first_command = obs[45:48].copy()
    next_obs, *_ = env.step(np.zeros(12))
    next_command = next_obs[45:48]

    assert np.allclose(first_command, env.obs_history[93:96])
    assert not np.allclose(first_command, next_command)


def test_verified_teacher_moves_target_model_forward_without_falling():
    import numpy as np

    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_env import DiskRobotWalkEnv

    config = WalkTaskConfig(
        action_repeat=5,
        max_episode_steps=350,
        reset_joint_noise=0.0,
        reset_height_noise=0.0,
        command_vx_min=0.15,
        command_vx_max=0.15,
        command_zero_probability=0.0,
        teacher_blend=1.0,
    )
    env = DiskRobotWalkEnv(config=config, seed=0)
    env.reset()
    start_x = float(env.data.qpos[0])
    terminated = False
    for _ in range(350):
        _, _, terminated, truncated, _ = env.step(np.zeros(config.action_size))
        if terminated or truncated:
            break

    assert not terminated
    # The legacy open-loop teacher is only a smoke baseline on the shorter active
    # structure candidate; it currently covers about 0.75 m in this rollout.
    assert float(env.data.qpos[0]) - start_x > 0.7
