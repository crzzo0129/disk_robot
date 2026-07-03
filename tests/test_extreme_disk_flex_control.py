import sys


def test_flex_control_imports_without_loading_mujoco():
    from scripts import control_extreme_disk_flex

    assert "mujoco" not in sys.modules
    assert control_extreme_disk_flex.PAIR_ACTUATORS["front"] == (
        "fl_hip_flex_act",
        "fr_hip_flex_act",
    )
    assert control_extreme_disk_flex.PAIR_ACTUATORS["rear"] == (
        "hl_hip_flex_act",
        "hr_hip_flex_act",
    )


def test_adjust_pair_sets_both_controls_from_pair_average():
    from scripts.control_extreme_disk_flex import adjust_pair

    ctrl = [0.0, -0.4, 0.9, 0.0, -0.6, 0.9]

    value = adjust_pair(ctrl, indices=(1, 4), delta=0.2, low=-1.6, high=1.6)

    assert round(value, 6) == -0.3
    assert ctrl[1] == ctrl[4] == -0.3


def test_adjust_pair_clips_to_range():
    from scripts.control_extreme_disk_flex import adjust_pair

    ctrl = [1.55, 1.45]

    value = adjust_pair(ctrl, indices=(0, 1), delta=0.3, low=-1.6, high=1.6)

    assert value == 1.6
    assert ctrl == [1.6, 1.6]


def test_command_queue_drains_keyboard_commands_in_order():
    from scripts.control_extreme_disk_flex import CommandQueue

    commands = CommandQueue()

    commands.push("front+")
    commands.push("reset")

    assert commands.drain() == ["front+", "reset"]
    assert commands.drain() == []
