"""Play a smooth interpolation between two robot keyframes."""
import argparse
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "disk_quadruped_extreme_train.xml"
PUPPER_XML_PATH = REPO_ROOT / "assets" / "pupper_v3_disk_visual.xml"
DEFAULT_FROM_KEYFRAME = "stand"
DEFAULT_TO_KEYFRAME = "folded"

KEY_HELP = """
Keyboard controls:
  Space : pause / resume
  R     : restart interpolation
  [ / ] : slow down / speed up
"""


class PlaybackState:
    def __init__(self, speed):
        self.speed = speed
        self.phase = 0.0
        self.paused = False
        self._lock = threading.Lock()
        self._commands = []

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
    parser.add_argument("--model", choices=("disk", "pupper"), default="disk")
    parser.add_argument("--xml-path", type=Path)
    parser.add_argument("--track-body")
    parser.add_argument("--from-keyframe")
    parser.add_argument("--to-keyframe")
    parser.add_argument("--period", type=float, default=4.0, help="Seconds for stand -> folded -> stand.")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args(argv)

    if args.model == "pupper":
        args.xml_path = args.xml_path or PUPPER_XML_PATH
        args.track_body = args.track_body or "base_link"
        args.from_keyframe = args.from_keyframe or "home"
    else:
        args.xml_path = args.xml_path or XML_PATH
        args.track_body = args.track_body or "disk_torso"
        args.from_keyframe = args.from_keyframe or DEFAULT_FROM_KEYFRAME
    args.to_keyframe = args.to_keyframe or DEFAULT_TO_KEYFRAME
    return args


def lerp_sequence(start, end, alpha):
    return [float(a) * (1.0 - alpha) + float(b) * alpha for a, b in zip(start, end)]


def triangle_phase(phase):
    phase = phase % 1.0
    if phase <= 0.5:
        return phase * 2.0
    return (1.0 - phase) * 2.0


def _keyframe_id(mujoco, model, name):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise ValueError(f"Keyframe not found: {name}")
    return key_id


def _keyframe_arrays(model, key_id):
    qpos = [float(value) for value in model.key_qpos[key_id]]
    ctrl = [float(value) for value in model.key_ctrl[key_id]]
    return qpos, ctrl


def main(argv=None):
    args = parse_args(argv)

    import mujoco
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(str(args.xml_path.resolve()))
    data = mujoco.MjData(model)
    from_id = _keyframe_id(mujoco, model, args.from_keyframe)
    to_id = _keyframe_id(mujoco, model, args.to_keyframe)
    from_qpos, from_ctrl = _keyframe_arrays(model, from_id)
    to_qpos, to_ctrl = _keyframe_arrays(model, to_id)
    state = PlaybackState(speed=args.speed)

    def apply_interpolated_pose(alpha):
        qpos = lerp_sequence(from_qpos, to_qpos, alpha)
        ctrl = lerp_sequence(from_ctrl, to_ctrl, alpha)
        data.qpos[:] = qpos
        data.ctrl[:] = ctrl
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def key_callback(keycode):
        if keycode == 32:
            state.push("toggle_pause")
            return
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "r":
            state.push("restart")
        elif key == "[":
            state.push("slower")
        elif key == "]":
            state.push("faster")

    def apply_command(command):
        if command == "toggle_pause":
            state.paused = not state.paused
            print(f"paused={state.paused}", flush=True)
        elif command == "restart":
            state.phase = 0.0
            print("restart interpolation", flush=True)
        elif command == "slower":
            state.speed = max(0.1, state.speed / 1.25)
            print(f"speed={state.speed:.3f}", flush=True)
        elif command == "faster":
            state.speed = min(5.0, state.speed * 1.25)
            print(f"speed={state.speed:.3f}", flush=True)

    apply_interpolated_pose(0.0)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.track_body)
    if torso_id < 0:
        raise ValueError(f"Tracking body not found: {args.track_body}")
    print(f"model={args.xml_path.resolve()}", flush=True)
    print(f"interpolating {args.from_keyframe} <-> {args.to_keyframe}", flush=True)
    print(KEY_HELP.strip(), flush=True)

    previous_time = time.perf_counter()
    with viewer.launch_passive(model, data, key_callback=key_callback) as window:
        window.cam.azimuth = 90
        window.cam.elevation = -8
        window.cam.distance = 1.4
        window.cam.lookat[:] = data.xpos[torso_id]
        while window.is_running():
            now = time.perf_counter()
            dt = now - previous_time
            previous_time = now
            with window.lock():
                for command in state.drain():
                    apply_command(command)
                if not state.paused:
                    state.phase = (state.phase + dt * state.speed / max(args.period, 1e-6)) % 1.0
                apply_interpolated_pose(triangle_phase(state.phase))
            window.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
