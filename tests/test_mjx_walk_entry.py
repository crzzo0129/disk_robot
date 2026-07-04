def test_mjx_walk_train_entry_imports_without_loading_jax_or_brax():
    import sys

    from scripts import mjx_train_walk

    assert "jax" not in sys.modules
    assert "brax" not in sys.modules
    assert mjx_train_walk.parse_args(["--steps", "123"]).steps == 123


def test_mjx_walk_train_entry_defaults_to_cloud_smoke_scale():
    from scripts.mjx_train_walk import parse_args

    args = parse_args([])

    assert args.envs == 128
    assert args.episode_length == 128
    assert args.mujoco_gl == "egl"
    assert args.xml_path.name == "disk_quadruped_extreme_train.xml"


def test_auto_mujoco_gl_prefers_egl_on_headless_linux():
    from disk_robot_mjx.pipeline import select_mujoco_gl_backend

    assert select_mujoco_gl_backend(environ={}, platform_name="linux") == "egl"


def test_mjx_walk_env_is_not_left_as_placeholder():
    source = open("disk_robot_mjx/brax_env.py", encoding="utf-8").read()

    assert "NotImplementedError" not in source


def test_mjx_walk_env_preserves_brax_wrapper_info_keys():
    source = open("disk_robot_mjx/brax_env.py", encoding="utf-8").read()

    assert 'info = {\n                "target_ctrl"' not in source
    assert "**state.info" in source
