# Disk Robot 常用指令

本文档假设当前目录是：

```bash
/inspire/hdd/project/leverage-robot/ky26210/disk_robot
```

并且已经激活 MJX/MuJoCo 环境，例如：

```bash
conda activate /inspire/hdd/project/leverage-robot/ky26210/conda_envs/mjx312
```

如果 import 找不到 `disk_robot`，在命令前加：

```bash
PYTHONPATH=.
```

---

## 1. 单元测试

### 跑 walking 相关测试

```bash
python -m pytest tests/test_walk_config.py tests/test_walk_reward.py tests/test_walk_smoke.py
```

### 跑全部测试

```bash
python -m pytest tests
```

---

## 2. Open-loop gait

`open_loop_gait.py` 用于直接运行手写 gait，不经过 RL。适合快速比较不同 gait 参数。

### 2.1 直接运行默认 trot

```bash
python scripts/open_loop_gait.py --duration 8 --mode trot
```

默认核心参数当前约为：

```text
frequency = 1.0
hip_amplitude = 0.10
knee_amplitude = 0.22
duty = 0.6
direction = -1
keyframe = walk_stand
xml = assets/disk_quadruped_extreme_train.xml
```

输出重点看：

```text
dx
dy
final_z
```

---

### 2.2 录制 open-loop 视频

使用 `--render` 保存 mp4：

```bash
python scripts/open_loop_gait.py \
  --mode trot \
  --render mjx_runs/open_loop_gait/trot.mp4 \
  --duration 8 \
  --frequency 1.0 \
  --duty 0.55 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1 \
  --fps 50 \
  --camera side_cam \
  --mujoco-gl egl
```

生成文件：

```text
mjx_runs/open_loop_gait/trot.mp4
```

---

### 2.3 打开 viewer 看 open-loop

```bash
python scripts/open_loop_gait.py \
  --duration 8 \
  --mode trot \
  --frequency 1.0 \
  --duty 0.55 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1 \
  --viewer
```

---

## 3. Open-loop 参数扫描常用命令

下面这些命令用于快速比较不同 gait。建议先看 `dx`、`dy/dx`、视频中身体扭动程度。

### 3.1 Trot，duty=0.55，abd=0

```bash
python scripts/open_loop_gait.py \
  --duration 8 \
  --mode trot \
  --frequency 1.0 \
  --duty 0.55 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1
```

录视频版本：

```bash
python scripts/open_loop_gait.py \
  --mode trot \
  --render mjx_runs/open_loop_gait/trot_d055_abd0.mp4 \
  --duration 8 \
  --frequency 1.0 \
  --duty 0.55 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1 \
  --fps 50 \
  --camera side_cam \
  --mujoco-gl egl
```

---

### 3.2 Trot，50% duty

接近标准对角 trot 的 50% stance/swing：

```bash
python scripts/open_loop_gait.py \
  --duration 8 \
  --mode trot \
  --frequency 1.0 \
  --duty 0.50 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1
```

录视频：

```bash
python scripts/open_loop_gait.py \
  --mode trot \
  --render mjx_runs/open_loop_gait/trot_d050_abd0.mp4 \
  --duration 8 \
  --frequency 1.0 \
  --duty 0.50 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1 \
  --fps 50 \
  --camera side_cam \
  --mujoco-gl egl
```

---

### 3.3 Bound

前腿一组、后腿一组。可能比 trot 少 yaw，但 pitch 可能更大。

```bash
python scripts/open_loop_gait.py \
  --duration 8 \
  --mode bound \
  --frequency 1.0 \
  --duty 0.55 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1
```

录视频：

```bash
python scripts/open_loop_gait.py \
  --mode bound \
  --render mjx_runs/open_loop_gait/bound_d055_abd0.mp4 \
  --duration 8 \
  --frequency 1.0 \
  --duty 0.55 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1 \
  --fps 50 \
  --camera side_cam \
  --mujoco-gl egl
```

---

### 3.4 Crawl

四条腿依次移动，低速可能更稳，但速度通常更慢。

```bash
python scripts/open_loop_gait.py \
  --duration 8 \
  --mode crawl \
  --frequency 1.0 \
  --duty 0.60 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1
```

录视频：

```bash
python scripts/open_loop_gait.py \
  --mode crawl \
  --render mjx_runs/open_loop_gait/crawl_d060_abd0.mp4 \
  --duration 8 \
  --frequency 1.0 \
  --duty 0.60 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1 \
  --fps 50 \
  --camera side_cam \
  --mujoco-gl egl
```

---

### 3.5 Pace

同侧腿一组，可能减少对角扭矩，但容易左右摇摆。

```bash
python scripts/open_loop_gait.py \
  --duration 8 \
  --mode pace \
  --frequency 1.0 \
  --duty 0.55 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1
```

录视频：

```bash
python scripts/open_loop_gait.py \
  --mode pace \
  --render mjx_runs/open_loop_gait/pace_d055_abd0.mp4 \
  --duration 8 \
  --frequency 1.0 \
  --duty 0.55 \
  --hip-stance-amplitude 0.10 \
  --hip-swing-amplitude 0.10 \
  --knee-lift-amplitude 0.22 \
  --abd-amplitude 0.0 \
  --direction -1 \
  --fps 50 \
  --camera side_cam \
  --mujoco-gl egl
```

---

## 4. walk_smoke：验证 residual=0 环境

`walk_smoke.py` 走的是训练环境接口：

```text
ctrl = open_loop_gait + residual_action
```

当 `--policy zero` 时就是：

```text
open_loop_gait + zero residual
```

---

### 4.1 residual=0，无 reset noise

用于和 open-loop baseline 对齐。

```bash
python scripts/walk_smoke.py \
  --steps 250 \
  --policy zero \
  --command-velocity 0.1 \
  --reset-joint-noise 0 \
  --reset-height-noise 0
```

输出重点看：

```text
dx
dy
dy_per_dx
heading_error
yaw_rate
lateral_velocity
forward_velocity
```

---

### 4.2 residual=0，默认 reset noise

更接近训练环境：

```bash
python scripts/walk_smoke.py \
  --steps 250 \
  --policy zero \
  --command-velocity 0.1
```

---

### 4.3 random residual

用于检查 residual action 通道是否容易把系统搞炸：

```bash
python scripts/walk_smoke.py \
  --steps 250 \
  --policy random \
  --command-velocity 0.1
```

---

## 5. MJX residual RL 训练

训练环境当前使用：

```text
ctrl = open_loop_gait + residual_scale * policy_action
```

默认会在训练结束后生成：

```text
<out>/final_policy.mp4
<out>/best_policy.mp4
```

---

### 5.1 快速 smoke 训练

```bash
python scripts/mjx_train_walk.py \
  --steps 10000 \
  --envs 128 \
  --episode-length 250 \
  --max-episode-steps 250 \
  --num-evals 2 \
  --num-eval-envs 64 \
  --command-velocity 0.1 \
  --out mjx_runs/residual_cmd01_smoke \
  --video-steps 500 \
  --video-camera tracking \
  --video-distance 2.0 \
  --video-azimuth 90 \
  --video-elevation -20
```

---

### 5.2 更长 episode，观察长期偏航

```bash
python scripts/mjx_train_walk.py \
  --steps 20000 \
  --envs 256 \
  --episode-length 500 \
  --max-episode-steps 500 \
  --num-evals 4 \
  --num-eval-envs 128 \
  --command-velocity 0.1 \
  --learning-rate 1e-4 \
  --entropy-cost 1e-3 \
  --penalty-yaw-rate 1.0 \
  --penalty-heading-error 2.0 \
  --penalty-ang-vel-xy 0.8 \
  --reward-lateral 1.0 \
  --out mjx_runs/residual_cmd01_turn_penalty_smoke \
  --video-steps 750 \
  --video-camera tracking \
  --video-distance 2.0 \
  --video-azimuth 90 \
  --video-elevation -20
```

---

### 5.3 正式一点的 residual 训练

```bash
python scripts/mjx_train_walk.py \
  --steps 50000 \
  --envs 256 \
  --episode-length 500 \
  --max-episode-steps 500 \
  --num-evals 5 \
  --num-eval-envs 128 \
  --command-velocity 0.1 \
  --learning-rate 1e-4 \
  --entropy-cost 1e-3 \
  --penalty-yaw-rate 1.0 \
  --penalty-heading-error 2.0 \
  --penalty-ang-vel-xy 0.8 \
  --reward-lateral 1.0 \
  --out mjx_runs/residual_cmd01_turn_penalty_v0 \
  --video-steps 750 \
  --video-camera tracking \
  --video-distance 2.0 \
  --video-azimuth 90 \
  --video-elevation -20
```

---

## 6. W&B 可选

默认训练会保存本地视频，不需要 W&B。

如果需要上传 W&B：

```bash
python scripts/mjx_train_walk.py \
  --steps 50000 \
  --envs 256 \
  --episode-length 500 \
  --max-episode-steps 500 \
  --num-evals 5 \
  --num-eval-envs 128 \
  --command-velocity 0.1 \
  --learning-rate 1e-4 \
  --entropy-cost 1e-3 \
  --penalty-yaw-rate 1.0 \
  --penalty-heading-error 2.0 \
  --penalty-ang-vel-xy 0.8 \
  --reward-lateral 1.0 \
  --out mjx_runs/residual_cmd01_wandb \
  --video-steps 750 \
  --video-camera tracking \
  --video-distance 2.0 \
  --video-azimuth 90 \
  --video-elevation -20 \
  --use-wandb \
  --wandb-mode online \
  --wandb-project disk_robot_walk \
  --wandb-run-name residual_cmd01_wandb
```

离线 W&B：

```bash
--use-wandb --wandb-mode offline
```

---

## 7. 常用指标怎么看

### Open-loop 输出

重点看：

```text
dx          前进距离
dy          侧向漂移
final_z     最终高度
```

可以手算：

```text
dy_per_dx = abs(dy) / max(abs(dx), 1e-6)
```

越小越直。

---

### walk_smoke 输出

重点看：

```text
dx
dy
dy_per_dx
heading_error
yaw_rate
lateral_velocity
forward_velocity
torso_height
disk_contacts
```

目标：

```text
dx > 0
dy_per_dx 小
heading_error 接近 0
yaw_rate 接近 0
lateral_velocity 接近 0
torso_height 稳定
disk_contacts = 0
```

---

### 训练日志

重点看：

```text
eval_reward
avg_episode_length
avg_forward_velocity
```

如果日志里有以下 metric，也重点看：

```text
eval/episode_reward_yaw
eval/episode_reward_heading
eval/episode_reward_lateral
eval/episode_yaw_rate
eval/episode_heading_error
eval/episode_lateral_velocity
```

目标不是单纯速度最快，而是：

```text
走得稳、走得直、episode 不提前结束、速度接近 command_velocity。
```
