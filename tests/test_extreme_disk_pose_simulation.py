import sys


def test_pose_simulation_imports_without_loading_mujoco():
    from scripts import simulate_extreme_disk_pose

    assert "mujoco" not in sys.modules
    assert simulate_extreme_disk_pose.DEFAULT_FROM_KEYFRAME == "walk_stand"
    assert simulate_extreme_disk_pose.DEFAULT_MIDDLE_KEYFRAME == "stand"
    assert simulate_extreme_disk_pose.DEFAULT_TO_KEYFRAME == "folded"


def test_lerp_sequence_clips_to_control_range():
    from scripts.simulate_extreme_disk_pose import lerp_sequence

    values = lerp_sequence([0.0, -2.0, 2.0], [2.0, -4.0, 4.0], 0.75, [(-1.0, 1.0), (-3.0, 3.0), (-2.5, 2.5)])

    assert values == [1.0, -3.0, 2.5]


def test_status_line_reports_physics_metrics():
    from scripts.simulate_extreme_disk_pose import format_status_line

    line = format_status_line(
        sim_time=1.25,
        stage="stand_to_folded",
        alpha=0.5,
        torso_x=0.12,
        torso_y=-0.03,
        torso_z=0.31,
        contact_count=4,
        disk_contact_count=1,
    )

    assert line == "t=1.25 stage=stand_to_folded alpha=0.50 x=0.120 y=-0.030 z=0.310 contacts=4 disk_contacts=1"


def test_staged_target_moves_through_walk_stand_stand_folded():
    from scripts.simulate_extreme_disk_pose import staged_target

    assert staged_target(0.25, switch_time=0.5, walk_to_stand_time=2.0, stand_hold_time=0.5, stand_to_folded_time=3.0) == (
        "walk_to_stand",
        0.0,
    )
    assert staged_target(1.5, switch_time=0.5, walk_to_stand_time=2.0, stand_hold_time=0.5, stand_to_folded_time=3.0) == (
        "walk_to_stand",
        0.5,
    )
    assert staged_target(2.75, switch_time=0.5, walk_to_stand_time=2.0, stand_hold_time=0.5, stand_to_folded_time=3.0) == (
        "stand_hold",
        0.0,
    )
    assert staged_target(4.5, switch_time=0.5, walk_to_stand_time=2.0, stand_hold_time=0.5, stand_to_folded_time=3.0) == (
        "stand_to_folded",
        0.5,
    )
    assert staged_target(6.0, switch_time=0.5, walk_to_stand_time=2.0, stand_hold_time=0.5, stand_to_folded_time=3.0) == (
        "folded",
        1.0,
    )


def test_step_target_smoothly_transitions_after_delay():
    from scripts.simulate_extreme_disk_pose import step_target_alpha

    assert step_target_alpha(0.49, switch_time=0.5, transition_time=2.0) == 0.0
    assert step_target_alpha(0.5, switch_time=0.5, transition_time=2.0) == 0.0
    assert step_target_alpha(1.5, switch_time=0.5, transition_time=2.0) == 0.5
    assert step_target_alpha(2.5, switch_time=0.5, transition_time=2.0) == 1.0
    assert step_target_alpha(10.0, switch_time=0.5, transition_time=2.0) == 1.0


def test_zero_transition_time_keeps_instant_switch_available():
    from scripts.simulate_extreme_disk_pose import step_target_alpha

    assert step_target_alpha(0.49, switch_time=0.5, transition_time=0.0) == 0.0
    assert step_target_alpha(0.5, switch_time=0.5, transition_time=0.0) == 1.0
