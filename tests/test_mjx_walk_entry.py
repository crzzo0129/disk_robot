def test_mjx_walk_train_entry_imports_without_loading_jax_or_brax():
    import sys

    from scripts import mjx_train_walk

    assert "jax" not in sys.modules
    assert "brax" not in sys.modules
    assert mjx_train_walk.parse_args(["--steps", "123"]).steps == 123


def test_mjx_walk_train_defaults_to_pupper_forward_profile():
    from scripts.mjx_train_walk import parse_args

    args = parse_args([])

    assert args.envs == 128
    assert args.episode_length == 128
    assert args.mujoco_gl == "egl"
    assert args.xml_path.name == "pupper_v3_disk_visual.xml"
    assert args.command_profile == "forward"


def test_mjx_walk_accepts_omnidirectional_command_ranges():
    from scripts.mjx_train_walk import parse_args

    args = parse_args(
        ["--command-profile", "omni", "--command-vx", "-0.1", "0.3", "--command-yaw", "-1", "1"]
    )

    assert args.command_profile == "omni"
    assert args.command_vx == [-0.1, 0.3]
    assert args.command_yaw == [-1.0, 1.0]


def test_mjx_walk_env_uses_shared_reward_and_no_runtime_gait():
    source = open("disk_robot_mjx/brax_env.py", encoding="utf-8").read()

    assert "reward_terms(jp" in source
    assert "gait_phase" not in source
    assert "open_loop" not in source
    assert "self.stand_q + self.action_scale * action" in source


def test_auto_mujoco_gl_prefers_egl_on_headless_linux():
    from disk_robot_mjx.pipeline import select_mujoco_gl_backend

    assert select_mujoco_gl_backend(environ={}, platform_name="linux") == "egl"
