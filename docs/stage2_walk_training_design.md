# Stage 2 Walk Training Design

## Goal

Stage 2 trains a basic flat-ground walking policy for the disk-body quadruped. The goal is not a polished gait. The goal is to validate the training interface that Stage 3 rolling and Stage 4 walk-roll switching will reuse: reset, observation, action semantics, reward terms, contact metrics, and cloud training entrypoints.

## Chosen Route

Use a two-layer route:

1. Local MuJoCo smoke environment.
   - Runs on the local machine.
   - Checks reset, step, observation size, reward, contact counting, and termination.
   - Does not require local PPO training.

2. Cloud-oriented MJX/Brax PPO entrypoint.
   - Mirrors the `robot_curl` workflow: local smoke first, large training on compute platform.
   - Keeps JAX, Brax, and MJX imports lazy so local help/tests do not initialize GPU runtimes.

## Environment Contract

- Model: `assets/disk_quadruped_extreme.xml`.
- Reset pose: `stand` keyframe plus small joint and height noise.
- Action: 12-dimensional joint target increment, one value for each existing actuator.
- Observation size: 51.
- Observation layout:
  - torso quaternion: 4
  - torso linear velocity: 3
  - torso angular velocity: 3
  - 12 joint positions
  - 12 joint velocities
  - previous action: 12
  - foot-ground contacts: 4
  - commanded forward velocity: 1
- Reward priorities:
  - match commanded forward velocity
  - discourage lateral velocity
  - keep torso upright and high enough
  - encourage reasonable foot-ground contact
  - penalize torso disk-ground contact during walking
  - penalize large and rapidly changing actions
- Termination:
  - torso height below threshold
  - torso upright measure below threshold
  - episode length reached

## Local Smoke Commands

```powershell
python -m scripts.walk_smoke --steps 100 --policy zero
python -m scripts.walk_smoke --steps 100 --policy random
```

Expected output includes observation size, action size, total reward, torso height, foot contact count, disk contact count, and forward velocity.

## Cloud Training Command

```bash
python -m scripts.mjx_train_walk --steps 10000 --envs 128 --episode-length 128
```

This is the first cloud smoke run. Larger runs should only start after the smoke run confirms MJX compilation, stepping, reward metrics, and parameter saving.

On GPU compute nodes the training entrypoint defaults to `--mujoco-gl egl`. If a CPU/headless node has a working OSMesa stack instead, override with `--mujoco-gl osmesa`.
