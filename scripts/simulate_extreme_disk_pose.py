"""Physically simulate the extreme disk model through staged pose transitions."""
import argparse
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = REPO_ROOT / "assets" / "disk_quadruped_extreme_train.xml"
DEFAULT_FROM_KEYFRAME = "walk_stand"
DEFAULT_MIDDLE_KEYFRAME = "stand"
DEFAULT_TO_KEYFRAME = "folded"

KEY_HELP = """
Keyboard controls:
  Space : pause / resume
  R     : reset simulation
"""


class PlaybackState:
    def __init__(self):
        self.paused = False
        self.switched = False
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
    parser.add_argument("--from-keyframe", default=DEFAULT_FROM_KEYFRAME)
    parser.add_argument("--middle-keyframe", default=DEFAULT_MIDDLE_KEYFRAME)
    parser.add_argument("--to-keyframe", default=DEFAULT_TO_KEYFRAME)
    parser.add_argument("--switch-time", type=float, default=0.5, help="Seconds to wait before starting walk_stand -> stand.")
    parser.add_argument("--walk-to-stand-time", type=float, default=2.0, help="Seconds to move from walk_stand to stand.")
    parser.add_argument("--stand-hold-time", type=float, default=0.5, help="Seconds to hold stand before folding.")
    parser.add_argument("--stand-to-folded-time", type=float, default=2.0, help="Seconds to move from stand to folded.")
    parser.add_argument("--headless", action="store_true", help="Run without the interactive viewer.")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps for --headless.")
    parser.add_argument("--status-interval", type=float, default=3, help="Seconds between status prints.")
    return parser.parse_args(argv)


def lerp_sequence(start, end, alpha, ranges=None):
    values = [float(a) * (1.0 - alpha) + float(b) * alpha for a, b in zip(start, end)]
    if ranges is None:
        return values
    return [min(max(value, float(low)), float(high)) for value, (low, high) in zip(values, ranges)]


def triangle_phase(phase):
    phase = phase % 1.0
    if phase <= 0.5:
        return phase * 2.0
    return (1.0 - phase) * 2.0


def transition_alpha(elapsed, duration):
    if elapsed < 0.0:
        return 0.0
    if duration <= 0.0:
        return 1.0
    return min(max(elapsed / duration, 0.0), 1.0)


def staged_target(sim_time, switch_time, walk_to_stand_time, stand_hold_time, stand_to_folded_time):
    first_elapsed = sim_time - switch_time
    first_alpha = transition_alpha(first_elapsed, walk_to_stand_time)
    if first_alpha < 1.0:
        return "walk_to_stand", first_alpha

    second_start = switch_time + max(walk_to_stand_time, 0.0) + max(stand_hold_time, 0.0)
    second_alpha = transition_alpha(sim_time - second_start, stand_to_folded_time)
    if second_alpha < 1.0:
        if sim_time < second_start:
            return "stand_hold", 0.0
        return "stand_to_folded", second_alpha
    return "folded", 1.0


def step_target_alpha(sim_time, switch_time, transition_time):
    return transition_alpha(sim_time - switch_time, transition_time)


def format_status_line(sim_time, stage, alpha, torso_x, torso_y, torso_z, contact_count, disk_contact_count):
    return (
        f"t={sim_time:.2f} stage={stage} alpha={alpha:.2f} "
        f"x={torso_x:.3f} y={torso_y:.3f} z={torso_z:.3f} "
        f"contacts={contact_count} disk_contacts={disk_contact_count}"
    )


def _keyframe_id(mujoco, model, name):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        raise ValueError(f"Keyframe not found: {name}")
    return key_id


def _body_id(mujoco, model, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Body not found: {name}")
    return body_id


def _geom_id(mujoco, model, name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        raise ValueError(f"Geom not found: {name}")
    return geom_id


def _keyframe_ctrl(model, key_id):
    return [float(value) for value in model.key_ctrl[key_id]]


def _control_ranges(model):
    return [(float(low), float(high)) for low, high in model.actuator_ctrlrange]


def _disk_contact_count(data, disk_geom_id):
    total = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 == disk_geom_id or contact.geom2 == disk_geom_id:
            total += 1
    return total


def _status_line(data, torso_id, disk_geom_id, stage, alpha):
    torso_pos = data.xpos[torso_id]
    return format_status_line(
        sim_time=float(data.time),
        stage=stage,
        alpha=alpha,
        torso_x=float(torso_pos[0]),
        torso_y=float(torso_pos[1]),
        torso_z=float(torso_pos[2]),
        contact_count=int(data.ncon),
        disk_contact_count=_disk_contact_count(data, disk_geom_id),
    )


def main(argv=None):
    args = parse_args(argv)

    import mujoco
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    from_id = _keyframe_id(mujoco, model, args.from_keyframe)
    middle_id = _keyframe_id(mujoco, model, args.middle_keyframe)
    to_id = _keyframe_id(mujoco, model, args.to_keyframe)
    from_ctrl = _keyframe_ctrl(model, from_id)
    middle_ctrl = _keyframe_ctrl(model, middle_id)
    to_ctrl = _keyframe_ctrl(model, to_id)
    ctrl_ranges = _control_ranges(model)
    torso_id = _body_id(mujoco, model, "disk_torso")
    disk_geom_id = _geom_id(mujoco, model, "torso_disk")
    state = PlaybackState()

    def reset_simulation():
        state.switched = False
        mujoco.mj_resetDataKeyframe(model, data, from_id)
        data.ctrl[:] = from_ctrl
        mujoco.mj_forward(model, data)

    def update_target():
        stage, alpha = staged_target(
            data.time,
            args.switch_time,
            args.walk_to_stand_time,
            args.stand_hold_time,
            args.stand_to_folded_time,
        )
        if not state.switched:
            if stage == "walk_to_stand":
                data.ctrl[:] = lerp_sequence(from_ctrl, middle_ctrl, alpha, ctrl_ranges)
            elif stage in ("stand_hold", "stand_to_folded"):
                fold_alpha = 0.0 if stage == "stand_hold" else alpha
                data.ctrl[:] = lerp_sequence(middle_ctrl, to_ctrl, fold_alpha, ctrl_ranges)
            else:
                data.ctrl[:] = lerp_sequence(middle_ctrl, to_ctrl, 1.0, ctrl_ranges)
        if not state.switched and stage == "folded":
            state.switched = True
            print(f"finished target transition to {args.to_keyframe} at t={data.time:.3f}", flush=True)
        return stage, alpha

    def step_physics():
        stage, alpha = staged_target(
            data.time,
            args.switch_time,
            args.walk_to_stand_time,
            args.stand_hold_time,
            args.stand_to_folded_time,
        )
        if not state.paused:
            stage, alpha = update_target()
            mujoco.mj_step(model, data)
        return stage, alpha

    def key_callback(keycode):
        if keycode == 32:
            state.push("toggle_pause")
            return
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if key == "r":
            state.push("reset")

    def apply_command(command):
        if command == "toggle_pause":
            state.paused = not state.paused
            print(f"paused={state.paused}", flush=True)
        elif command == "reset":
            reset_simulation()
            print("reset simulation", flush=True)

    reset_simulation()
    print(
        f"physically simulating {args.from_keyframe} -> {args.middle_keyframe} -> {args.to_keyframe}",
        flush=True,
    )
    print(
        f"switch_time={args.switch_time:.3f}s "
        f"walk_to_stand_time={args.walk_to_stand_time:.3f}s "
        f"stand_hold_time={args.stand_hold_time:.3f}s "
        f"stand_to_folded_time={args.stand_to_folded_time:.3f}s; "
        "after alpha reaches 1.0 the target stays fixed.",
        flush=True,
    )

    if args.headless:
        last_status_time = -float("inf")
        stage = "walk_to_stand"
        alpha = 0.0
        for _ in range(max(args.steps, 0)):
            stage, alpha = step_physics()
            if data.time - last_status_time >= args.status_interval:
                print(_status_line(data, torso_id, disk_geom_id, stage, alpha), flush=True)
                last_status_time = data.time
        print(_status_line(data, torso_id, disk_geom_id, stage, alpha), flush=True)
        return

    print(KEY_HELP.strip(), flush=True)
    print(f"status_interval={args.status_interval:.3f}s; close the viewer window to exit.", flush=True)

    last_status_time = -float("inf")
    with viewer.launch_passive(model, data, key_callback=key_callback) as window:
        window.cam.azimuth = 90
        window.cam.elevation = -8
        window.cam.distance = 1.4
        window.cam.lookat[:] = data.xpos[torso_id]
        while window.is_running():
            with window.lock():
                for command in state.drain():
                    apply_command(command)
                stage, alpha = step_physics()
                window.cam.lookat[:] = data.xpos[torso_id]
                if data.time - last_status_time >= args.status_interval:
                    print(_status_line(data, torso_id, disk_geom_id, stage, alpha), flush=True)
                    last_status_time = data.time
            window.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
