# Disk Robot 常用命令

> 当前训练定义见 [forward_teacher_student.md](forward_teacher_student.md)。本页只记录可运行入口，不定义算法。

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

## IK 与结构检查

```powershell
python3.12 scripts\view_ik_gait.py --training-reference --neutral-pose model --duration 0
python3.12 scripts\view_ik_gait.py --xml assets\pupper_v3_disk_structure_candidate.xml --training-reference --neutral-pose model --mode trot --target-speed 0.08 --kp 10 --kd 0.4 --torque-limit 3 --duration 0
python -m scripts.sweep_structure_variants
```

## 当前 Teacher-Student 训练

云端 smoke：

```bash
python -m scripts.train_forward_teacher_student --smoke --out mjx_runs/forward_008_smoke
```

Smoke 仅验证整条软件链路。Student 验收失败不表示代码报错，先检查 `stage=ik_baseline` 和 `stage=teacher_acceptance`。

结构和 IK baseline 经可视化确认后，运行正式 `0.08 m/s` 前进训练：

```bash
python -m scripts.train_forward_teacher_student --out mjx_runs/forward_008_v1 --teacher-evals 21 --strict-acceptance
```

单独评估 Student：

```bash
python -m scripts.evaluate_forward_student mjx_runs/forward_008_v1/student_policy.npz
```

旧的 `scripts.mjx_train_walk`、`teacher_blend` 退火和 `0.15 m/s` 命令仍可用于历史实验，但不再是当前 pipeline 的验收入口。

## 云端提示

- 默认 MuJoCo GL 后端使用 `egl`。
- 仅在 OSMesa 配置完整的机器上使用 `--mujoco-gl osmesa`。
- 长训练前先运行单元测试和短 MJX 冒烟，并使用新的输出目录保存配置和评估结果。
