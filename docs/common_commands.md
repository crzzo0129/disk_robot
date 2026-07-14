# Disk Robot 常用命令

> 当前训练定义见 [omnidirectional_training_pipeline.md](omnidirectional_training_pipeline.md)。本页只记录可运行入口，不定义算法。

以下命令默认从 `disk_robot/` 目录执行。

## 安装与测试

```powershell
python -m pip install -r requirements.txt
python -m pytest tests
```

MJX 环境另行安装：

```powershell
python -m pip install -r requirements-mjx.txt
```

## XML 与姿态检查

```powershell
python -m scripts.diagnose_extreme_disk_pose --keyframe all
python -m scripts.view_extreme_disk_pose --keyframe stand
python -m scripts.view_extreme_disk_pose --keyframe folded
python -m scripts.control_extreme_disk_flex --keyframe folded
```

## 姿态转换与物理仿真

```powershell
python -m scripts.interpolate_extreme_disk_pose
python -m scripts.simulate_extreme_disk_pose
python -m scripts.simulate_extreme_disk_pose --headless --steps 1000
```

## 行走环境诊断

```powershell
python -m scripts.walk_smoke --steps 100 --policy zero
python -m scripts.walk_smoke --steps 100 --policy random
python -m scripts.mjx_train_walk --help
```

迁移期 MJX 冒烟：

```powershell
python -m scripts.mjx_train_walk --steps 10000 --envs 128 --episode-length 128 --command-profile forward
```

`10k` steps 只能检查编译、rollout 和日志链路。正式训练依次使用：

```powershell
python -m scripts.mjx_train_walk --steps 5000000 --envs 2048 --episode-length 500 --command-profile forward --out mjx_runs/forward
python -m scripts.mjx_train_walk --steps 20000000 --envs 2048 --episode-length 500 --command-profile omni --out mjx_runs/omni
python -m scripts.mjx_train_walk --steps 30000000 --envs 2048 --episode-length 500 --command-profile full --out mjx_runs/full
```

当前 Brax 入口会分别启动 run；在没有可靠 optimizer-state 恢复前，不应把它描述为连续 curriculum。先使用 `forward` 验证环境可学，再决定是否增加 checkpoint/optimizer 连续迁移。

## Teacher 引导训练

第一阶段只学习已验证 teacher，固定 `vx=0.15 m/s`：

```bash
python -m scripts.mjx_train_walk \
  --steps 1000000 --envs 1024 --episode-length 500 \
  --command-profile forward --command-vx 0.15 0.15 \
  --teacher-blend 1.0 --reward-teacher-imitation 1.0 \
  --out mjx_runs/teacher_stage1
```

第二阶段让策略承担一半控制：

```bash
python -m scripts.mjx_train_walk \
  --steps 2000000 --envs 1024 --episode-length 500 \
  --command-profile forward --command-vx 0.15 0.15 \
  --teacher-blend 0.5 --reward-teacher-imitation 0.3 \
  --restore-checkpoint mjx_runs/teacher_stage1/ppo_checkpoint \
  --out mjx_runs/teacher_stage2
```

第三阶段完全关闭 teacher：

```bash
python -m scripts.mjx_train_walk \
  --steps 3000000 --envs 1024 --episode-length 500 \
  --command-profile forward --command-vx 0.15 0.15 \
  --teacher-blend 0.0 --reward-teacher-imitation 0.0 \
  --restore-checkpoint mjx_runs/teacher_stage2/ppo_checkpoint \
  --out mjx_runs/teacher_stage3_pure
```

只有第三阶段视频仍能前进，才说明策略真正接管。脚本会自动选择 `ppo_checkpoint/` 下编号最大的 checkpoint。若该目录没有生成，说明云端 Brax 版本不支持官方 checkpoint API，应先升级或检查启动日志。

## 云端提示

- 默认 MuJoCo GL 后端使用 `egl`。
- 仅在 OSMesa 配置完整的机器上使用 `--mujoco-gl osmesa`。
- 长训练前先运行单元测试和短 MJX 冒烟，并保存配置、commit 和评估 command 网格。
