import sys


def test_pose_interpolation_imports_without_loading_mujoco():
    from scripts import interpolate_extreme_disk_pose

    assert "mujoco" not in sys.modules
    assert interpolate_extreme_disk_pose.DEFAULT_FROM_KEYFRAME == "stand"
    assert interpolate_extreme_disk_pose.DEFAULT_TO_KEYFRAME == "folded"


def test_triangle_phase_moves_forward_then_backward():
    from scripts.interpolate_extreme_disk_pose import triangle_phase

    assert triangle_phase(0.0) == 0.0
    assert triangle_phase(0.25) == 0.5
    assert triangle_phase(0.5) == 1.0
    assert triangle_phase(0.75) == 0.5
    assert triangle_phase(1.0) == 0.0


def test_lerp_sequence_interpolates_numeric_values():
    from scripts.interpolate_extreme_disk_pose import lerp_sequence

    assert lerp_sequence([0.0, 2.0, -2.0], [2.0, 4.0, 2.0], 0.25) == [0.5, 2.5, -1.0]
