def test_walk_env_imports_without_loading_mujoco():
    import sys

    from disk_robot.walk_env import DEFAULT_XML, DiskRobotWalkEnv

    assert "mujoco" not in sys.modules
    assert DEFAULT_XML.name == "disk_quadruped_extreme.xml"
    assert DiskRobotWalkEnv.__name__ == "DiskRobotWalkEnv"


def test_walk_env_reset_and_step_smoke():
    import numpy as np

    from disk_robot.walk_config import WalkTaskConfig
    from disk_robot.walk_env import DiskRobotWalkEnv

    config = WalkTaskConfig(action_repeat=2)
    env = DiskRobotWalkEnv(config=config, seed=1)

    obs, info = env.reset()
    assert obs.shape == (config.observation_size,)
    assert info["torso_height"] > 0.0

    obs, reward, terminated, truncated, info = env.step(np.zeros(config.action_size))
    assert obs.shape == (config.observation_size,)
    assert isinstance(float(reward), float)
    assert terminated in (False, True)
    assert truncated in (False, True)
    assert "foot_contact_count" in info
