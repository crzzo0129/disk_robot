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
    assert args.xml_path.name == "pupper_v3_disk_structure_candidate.xml"
    assert args.command_profile == "forward"


def test_mjx_walk_accepts_omnidirectional_command_ranges():
    from scripts.mjx_train_walk import parse_args

    args = parse_args(
        ["--command-profile", "omni", "--command-vx", "-0.1", "0.3", "--command-yaw", "-1", "1"]
    )

    assert args.command_profile == "omni"
    assert args.command_vx == [-0.1, 0.3]
    assert args.command_yaw == [-1.0, 1.0]


def test_mjx_walk_accepts_teacher_fade_and_checkpoint_restore():
    from scripts.mjx_train_walk import parse_args

    args = parse_args(
        [
            "--teacher-blend",
            "0.5",
            "--reward-teacher-imitation",
            "0.3",
            "--restore-checkpoint",
            "previous_checkpoint",
        ]
    )

    assert args.teacher_blend == 0.5
    assert args.reward_teacher_imitation == 0.3
    assert args.restore_checkpoint.name == "previous_checkpoint"


def test_mjx_walk_env_uses_shared_reward_and_no_runtime_gait():
    source = open("disk_robot_mjx/brax_env.py", encoding="utf-8").read()

    assert "reward_terms(jp" in source
    assert "gait_phase" not in source
    assert "use_open_loop_gait" not in source
    assert "self.stand_q + self.action_scale * blended_action" in source


def test_auto_mujoco_gl_prefers_egl_on_headless_linux():
    from disk_robot_mjx.pipeline import select_mujoco_gl_backend

    assert select_mujoco_gl_backend(environ={}, platform_name="linux") == "egl"


def test_restore_checkpoint_parent_selects_latest_numbered_child(tmp_path):
    from scripts.mjx_train_walk import _resolve_restore_checkpoint

    (tmp_path / "000000001000").mkdir()
    latest = tmp_path / "000000010000"
    latest.mkdir()

    assert _resolve_restore_checkpoint(tmp_path) == latest


def test_restore_checkpoint_path_is_made_absolute(tmp_path, monkeypatch):
    from scripts.mjx_train_walk import _resolve_restore_checkpoint

    monkeypatch.chdir(tmp_path)

    assert _resolve_restore_checkpoint(tmp_path.name).is_absolute()
