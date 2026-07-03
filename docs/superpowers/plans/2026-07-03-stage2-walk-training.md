# Stage 2 Walk Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local MuJoCo smoke walk task and a cloud-oriented MJX/Brax PPO training entrypoint for the disk robot.

**Architecture:** Put reusable task code in `disk_robot/`, cloud MJX helpers in `disk_robot_mjx/`, and runnable entrypoints in `scripts/`. The MuJoCo smoke env validates reset, step, observation, reward, and contact metrics locally; the MJX env mirrors the same task contract for compute-platform PPO.

**Tech Stack:** Python, NumPy, MuJoCo, MJX, Brax PPO, PowerShell/Bash.

---

### Task 1: Task Config And Reward

**Files:**
- Create: `disk_robot/walk_config.py`
- Create: `disk_robot/walk_reward.py`
- Test: `tests/test_walk_config.py`
- Test: `tests/test_walk_reward.py`

- [x] **Step 1: Add tests for 12-DOF action size and reward direction.**
- [x] **Step 2: Implement `WalkTaskConfig` and `compute_walk_reward`.**
- [x] **Step 3: Verify reward prefers matching forward velocity and penalizes disk contact.**

### Task 2: Local MuJoCo Smoke Environment

**Files:**
- Create: `disk_robot/walk_env.py`
- Create: `scripts/walk_smoke.py`
- Test: `tests/test_walk_smoke.py`

- [x] **Step 1: Add import and reset/step smoke tests.**
- [x] **Step 2: Implement `DiskRobotWalkEnv` with lazy MuJoCo import.**
- [x] **Step 3: Add CLI smoke script for zero and random actions.**
- [x] **Step 4: Run `python -m scripts.walk_smoke --steps 50 --policy zero`.**
- [x] **Step 5: Run `python -m scripts.walk_smoke --steps 50 --policy random --seed 2`.**

### Task 3: Cloud MJX/Brax Entrypoint

**Files:**
- Create: `disk_robot_mjx/pipeline.py`
- Create: `disk_robot_mjx/brax_env.py`
- Create: `scripts/mjx_train_walk.py`
- Test: `tests/test_mjx_walk_entry.py`

- [x] **Step 1: Add tests that command parsing does not import JAX/Brax.**
- [x] **Step 2: Implement cloud runtime helpers and train argument parsing.**
- [x] **Step 3: Implement first MJX/Brax walk environment matching the MuJoCo task contract.**
- [x] **Step 4: Keep heavy training dependencies lazy for local tests.**

### Task 4: Documentation

**Files:**
- Create: `docs/stage2_walk_training_design.md`
- Modify: `README.md`
- Modify: `docs/todo_extreme_disk_quadruped.md`

- [x] **Step 1: Document local smoke commands.**
- [x] **Step 2: Document cloud smoke training command.**
- [x] **Step 3: Mark Stage 2 local smoke infrastructure complete.**
