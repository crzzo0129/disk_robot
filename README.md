                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        # Disk Robot

MuJoCo model and phase-1 tools for an extreme disk-body quadruped. The robot keeps a normal quadruped layout, while the torso is a thin disk/cylinder: side view is close to a circle, front view remains narrow.

## Project Layout

- `assets/`: MJCF/XML model assets
- `scripts/`: viewer, keyboard control, pose interpolation, and diagnostics
- `tests/`: lightweight regression tests for the phase-1 model and tools
- `docs/`: design notes, TODOs, and phase handoff records

## Quick Start

From this project root:

```powershell
python -m pip install -r requirements.txt
python -m scripts.diagnose_extreme_disk_pose --keyframe all
python -m scripts.view_extreme_disk_pose --keyframe stand
python -m scripts.view_extreme_disk_pose --keyframe folded
python -m scripts.control_extreme_disk_flex --keyframe folded
python -m scripts.interpolate_extreme_disk_pose
python -m scripts.walk_smoke --steps 100 --policy zero
python -m scripts.walk_smoke --steps 100 --policy random
```

The current phase-1 baseline is `assets/disk_quadruped_extreme.xml` with two manually calibrated keyframes:

- `stand`: temporary standard standing pose.
- `folded`: temporary folded pose for rolling preparation.

The XML keeps 12 independent position actuators. Paired front/rear flex control is implemented only in Python scripts, not as physical actuator coupling in the model.

## Current Notes

- `stand` first frame has four foot-ground contacts, but the foot spheres start with penetration into the floor contact.
- `folded` first frame places the disk torso on the ground and folds the feet upward, with foot-torso internal contacts to watch during later collision design.
- Before walking training, see `docs/todo_extreme_disk_quadruped.md` and `docs/design_extreme_disk_quadruped.md`.

## Stage 2 Walk Training

Local work is limited to smoke tests. Cloud training starts from:

```bash
python -m pip install -r requirements-mjx.txt
python -m scripts.mjx_train_walk --steps 10000 --envs 128 --episode-length 128
```

The first cloud run should be treated as an MJX compilation and task-metric smoke test before longer PPO runs.
The default MuJoCo GL backend for cloud training is `egl`; use `--mujoco-gl osmesa` only on machines with a working OSMesa stack.
