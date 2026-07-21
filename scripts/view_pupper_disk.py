"""View the Pupper V3 model with a disk torso."""
import argparse
import time
from pathlib import Path

from disk_robot.model_paths import ACTIVE_MODEL_XML

XML_PATH = ACTIVE_MODEL_XML


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=XML_PATH)
    parser.add_argument("--keyframe", default="folded")
    parser.add_argument(
        "--passive",
        action="store_true",
        help="Use passive viewer mode for scripts that want to step MuJoCo themselves.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import mujoco
    from mujoco import viewer

    xml_path = args.xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if key_id < 0:
        raise ValueError(f"Keyframe not found: {args.keyframe}")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    print(f"model={xml_path} keyframe={args.keyframe}")
    if not args.passive:
        viewer.launch(model, data)
        return

    with viewer.launch_passive(model, data) as window:
        window.cam.azimuth = 120
        window.cam.elevation = -15
        window.cam.distance = 1.0
        window.cam.lookat[:] = data.xpos[torso_id]
        while window.is_running():
            mujoco.mj_step(model, data)
            window.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
