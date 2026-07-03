"""View stand or folded keyframes for the extreme disk quadruped model."""
import argparse
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "disk_quadruped_extreme.xml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe", choices=["stand", "folded"], default="stand")
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

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if key_id < 0:
        raise ValueError(f"Keyframe not found: {args.keyframe}")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "disk_torso")

    print(f"model={XML_PATH} keyframe={args.keyframe}")
    if not args.passive:
        viewer.launch(model, data)
        return

    with viewer.launch_passive(model, data) as window:
        window.cam.azimuth = 90
        window.cam.elevation = -8
        window.cam.distance = 1.4
        window.cam.lookat[:] = data.xpos[torso_id]
        while window.is_running():
            mujoco.mj_step(model, data)
            window.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
