import importlib


def test_pose_diagnostics_imports_without_loading_mujoco():
    module = importlib.import_module("scripts.diagnose_extreme_disk_pose")

    assert module.DEFAULT_XML.name == "disk_quadruped_extreme.xml"


def test_format_scalar_rounds_named_values():
    module = importlib.import_module("scripts.diagnose_extreme_disk_pose")

    assert module.format_scalar("torso_z", 0.321987) == "torso_z=0.3220"


def test_keyframe_selection_expands_all():
    module = importlib.import_module("scripts.diagnose_extreme_disk_pose")

    assert module.expand_keyframes("all") == ("stand", "folded")
    assert module.expand_keyframes("stand") == ("stand",)
