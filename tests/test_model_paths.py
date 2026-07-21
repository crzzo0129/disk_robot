from disk_robot.model_paths import ACTIVE_MODEL_XML, BASE_MODEL_XML


def test_canonical_model_paths_are_distinct_and_exist():
    assert ACTIVE_MODEL_XML.name == "pupper_v3_disk_structure_candidate.xml"
    assert BASE_MODEL_XML.name == "pupper_v3_disk_visual.xml"
    assert ACTIVE_MODEL_XML != BASE_MODEL_XML
    assert ACTIVE_MODEL_XML.is_file()
    assert BASE_MODEL_XML.is_file()


def test_runtime_defaults_use_the_active_model():
    from disk_robot.walk_env import DEFAULT_XML as cpu_default
    from disk_robot_mjx.brax_env import TRAIN_XML_PATH as mjx_default
    from disk_robot_mjx.teacher_student_env import DEFAULT_XML as teacher_student_default

    assert cpu_default == ACTIVE_MODEL_XML
    assert mjx_default == ACTIVE_MODEL_XML
    assert teacher_student_default == ACTIVE_MODEL_XML
