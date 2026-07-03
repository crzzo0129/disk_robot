"""Keyboard control for paired hip flex actuators on the extreme disk model."""
import argparse
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "disk_quadruped_extreme.xml"

PAIR_ACTUATORS = {
    "front": ("fl_hip_flex_act", "fr_hip_flex_act"),
    "rear": ("hl_hip_flex_act", "hr_hip_flex_act"),
}

KEY_HELP = """
Keyboard controls:
  Q / A : front hip_flex pair + / -
  W / S : rear  hip_flex pair + / -
  E / D : all   hip_flex pairs + / -
  R     : reset to selected keyframe
"""


class CommandQueue:
    def __init__(self):
        self._commands = []
        self._lock = threading.Lock()

    def push(self, command):
        with self._lock:
            self._commands.append(command)

    def drain(self):
        with self._lock:
            commands = self._commands
            self._commands = []
        return commands


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframe", choices=["stand", "folded"], default="folded")
    parser.add_argument("--step", type=float, default=0.5, help="Angle delta per key press, in radians.")
    return parser.parse_args(argv)


def adjust_pair(ctrl, indices, delta, low, high):
    current = sum(float(ctrl[index]) for index in indices) / len(indices)
    target = min(max(current + delta, low), high)
    for index in indices:
        ctrl[index] = target
    return target


def _actuator_id(mujoco, model, name):
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id < 0:
        raise ValueError(f"Actuator not found: {name}")
    return actuator_id


def _pair_indices_and_range(mujoco, model, pair_name):
    indices = tuple(_actuator_id(mujoco, model, name) for name in PAIR_ACTUATORS[pair_name])
    low = max(float(model.actuator_ctrlrange[index][0]) for index in indices)
    high = min(float(model.actuator_ctrlrange[index][1]) for index in indices)
    return indices, low, high


def main(argv=None):
    args = parse_args(argv)

    import mujoco
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if key_id < 0:
        raise ValueError(f"Keyframe not found: {args.keyframe}")

    front_indices, front_low, front_high = _pair_indices_and_range(mujoco, model, "front")
    rear_indices, rear_low, rear_high = _pair_indices_and_range(mujoco, model, "rear")
    commands = CommandQueue()

    def reset_keyframe():
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        print(f"reset keyframe={args.keyframe}", flush=True)

    def adjust_front(delta):
        value = adjust_pair(data.ctrl, front_indices, delta, front_low, front_high)
        print(f"front hip_flex target={value:.3f}", flush=True)

    def adjust_rear(delta):
        value = adjust_pair(data.ctrl, rear_indices, delta, rear_low, rear_high)
        print(f"rear hip_flex target={value:.3f}", flush=True)

    def key_callback(keycode):
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "q":
            commands.push("front+")
        elif key == "a":
            commands.push("front-")
        elif key == "w":
            commands.push("rear+")
        elif key == "s":
            commands.push("rear-")
        elif key == "e":
            commands.push("all+")
        elif key == "d":
            commands.push("all-")
        elif key == "r":
            commands.push("reset")

    def apply_command(command):
        if command == "front+":
            adjust_front(args.step)
        elif command == "front-":
            adjust_front(-args.step)
        elif command == "rear+":
            adjust_rear(args.step)
        elif command == "rear-":
            adjust_rear(-args.step)
        elif command == "all+":
            adjust_front(args.step)
            adjust_rear(args.step)
        elif command == "all-":
            adjust_front(-args.step)
            adjust_rear(-args.step)
        elif command == "reset":
            reset_keyframe()

    reset_keyframe()
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "disk_torso")
    print(KEY_HELP.strip(), flush=True)
    print(f"step={args.step:.3f} rad; close the viewer window to exit.", flush=True)

    with viewer.launch_passive(model, data, key_callback=key_callback) as window:
        window.cam.azimuth = 90
        window.cam.elevation = -8
        window.cam.distance = 1.4
        window.cam.lookat[:] = data.xpos[torso_id]
        while window.is_running():
            with window.lock():
                for command in commands.drain():
                    apply_command(command)
                mujoco.mj_step(model, data)
            window.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
