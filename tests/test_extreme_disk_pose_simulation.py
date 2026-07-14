import sys
import xml.etree.ElementTree as ET


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


def test_rear_push_roll_target_folds_before_pushing():
    from scripts.simulate_extreme_disk_pose import rear_push_roll_target

    target = lambda sim_time: rear_push_roll_target(sim_time, 0.5, 0.2, 0.4, 0.2, 0.1, 0.25, 0.4)

    assert target(0.5) == ("front_fold", 0.0)
    assert target(0.6)[0] == "front_fold"
    assert abs(target(0.6)[1] - 0.5) < 1e-12
    assert target(0.9)[0] == "rear_fold"
    assert abs(target(0.9)[1] - 0.5) < 1e-12
    assert target(1.2) == ("folded_hold", 0.0)
    assert target(1.35)[0] == "folded_to_push"
    assert target(1.5) == ("push_hold", 0.0)
    assert target(1.8)[0] == "push_to_folded"
    assert target(2.1) == ("rolling", 1.0)


def test_zero_transition_time_keeps_instant_switch_available():
    from scripts.simulate_extreme_disk_pose import step_target_alpha

    assert step_target_alpha(0.49, switch_time=0.5, transition_time=0.0) == 0.0
    assert step_target_alpha(0.5, switch_time=0.5, transition_time=0.0) == 1.0


def test_pupper_preset_selects_home_and_pupper_geometry():
    from scripts import simulate_extreme_disk_pose

    args = simulate_extreme_disk_pose.parse_args(["--model", "pupper"])

    assert args.xml_path == simulate_extreme_disk_pose.PUPPER_XML_PATH
    assert args.from_keyframe == "home"
    assert args.middle_keyframe == "home"
    assert args.to_keyframe == "folded"
    assert args.track_body == "base_link"
    assert args.disk_geom == "base_disk_collision"
    assert args.walk_to_stand_time == 0.0


def test_rear_push_roll_preset_selects_motion_and_runtime_strength():
    from scripts import simulate_extreme_disk_pose

    args = simulate_extreme_disk_pose.parse_args(["--motion", "rear-push-roll"])

    assert args.model == "pupper"
    assert args.from_keyframe == "home"
    assert args.middle_keyframe == "rear_push"
    assert args.to_keyframe == "folded"
    assert args.switch_time == 0.5
    assert args.front_fold_time == 0.18
    assert args.rear_fold_time == 0.55
    assert args.folded_hold_time == 0.1
    assert args.walk_to_stand_time == 0.2
    assert args.stand_hold_time == 0.15
    assert args.stand_to_folded_time == 0.18
    assert args.kp == 60.0
    assert args.kd == 1.0
    assert args.force_limit == 6.0


def test_rear_push_keeps_front_legs_folded():
    from scripts import simulate_extreme_disk_pose

    root = ET.parse(simulate_extreme_disk_pose.PUPPER_XML_PATH).getroot()
    folded = root.find("./keyframe/key[@name='folded']")
    rear_push = root.find("./keyframe/key[@name='rear_push']")
    folded_ctrl = [float(value) for value in folded.attrib["ctrl"].split()]
    rear_push_ctrl = [float(value) for value in rear_push.attrib["ctrl"].split()]

    assert rear_push_ctrl[:6] == folded_ctrl[:6]
    assert rear_push_ctrl[6:] != folded_ctrl[6:]


def test_pupper_home_starts_above_the_contacting_pose():
    from scripts import simulate_extreme_disk_pose

    root = ET.parse(simulate_extreme_disk_pose.PUPPER_XML_PATH).getroot()
    home = root.find("./keyframe/key[@name='home']")
    home_qpos = [float(value) for value in home.attrib["qpos"].split()]

    assert home_qpos[2] == 0.30
