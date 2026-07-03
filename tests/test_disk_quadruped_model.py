from pathlib import Path
import xml.etree.ElementTree as ET


XML_PATH = Path("assets/disk_quadruped_extreme.xml")


def test_extreme_disk_model_declares_disk_body_and_keyframes():
    assert XML_PATH.exists()

    root = ET.parse(XML_PATH).getroot()

    torso_body = root.find(".//body[@name='disk_torso']")
    torso_geom = root.find(".//geom[@name='torso_disk']")
    assert torso_body is not None
    assert torso_geom is not None
    assert torso_geom.get("type") == "cylinder"

    radius, half_thickness = [float(value) for value in torso_geom.get("size").split()[:2]]
    assert 0.18 <= radius <= 0.30
    assert 0.02 <= half_thickness <= 0.08

    key_names = {node.get("name") for node in root.findall("./keyframe/key")}
    assert {"stand", "folded"}.issubset(key_names)


def test_extreme_disk_model_has_four_symmetric_three_dof_legs():
    root = ET.parse(XML_PATH).getroot()
    joint_names = {node.get("name") for node in root.findall(".//joint")}
    actuator_joints = {node.get("joint") for node in root.findall("./actuator/*")}

    expected_joints = {
        "fl_hip_abd",
        "fl_hip_flex",
        "fl_knee",
        "fr_hip_abd",
        "fr_hip_flex",
        "fr_knee",
        "hl_hip_abd",
        "hl_hip_flex",
        "hl_knee",
        "hr_hip_abd",
        "hr_hip_flex",
        "hr_knee",
    }

    assert expected_joints.issubset(joint_names)
    assert expected_joints == actuator_joints

    anchor_bodies = {node.get("name") for node in root.findall(".//body") if node.get("name", "").endswith("_hip_anchor")}
    assert anchor_bodies == {
        "fl_hip_anchor",
        "fr_hip_anchor",
        "hl_hip_anchor",
        "hr_hip_anchor",
    }


def test_extreme_disk_actuators_have_viewer_control_ranges():
    root = ET.parse(XML_PATH).getroot()
    actuator_nodes = root.findall("./actuator/*")

    assert len(actuator_nodes) == 12
    for node in actuator_nodes:
        ctrlrange = node.get("ctrlrange")
        assert ctrlrange is not None, node.get("name")
        low, high = [float(value) for value in ctrlrange.split()]
        assert low < high


def test_extreme_disk_pose_viewer_imports_without_loading_mujoco():
    import sys

    from scripts import view_extreme_disk_pose

    assert "mujoco" not in sys.modules
    assert view_extreme_disk_pose.XML_PATH == XML_PATH.resolve()
    assert view_extreme_disk_pose.parse_args(["--keyframe", "folded"]).keyframe == "folded"
