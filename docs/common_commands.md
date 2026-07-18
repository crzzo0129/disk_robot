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

## 当前 Teacher-Student 训练与诊断

冻结的 accepted Teacher 是 `mjx_runs/teacher_t2a_seed0`，accepted Student 是
`mjx_runs/student_t8_phase_bc_no_previous_action_seed0`。不要覆盖这两个目录。

Teacher 全链路 smoke：

```bash
python -m scripts.train_forward_teacher_student --smoke --out mjx_runs/forward_008_smoke
```

Smoke 仅验证软件链路。验收失败不等于代码报错，先检查每个 stage 的明确结果。

T8 no-previous-action Student smoke：

```bash
python -m scripts.distill_phase_student_no_previous_action --teacher-run mjx_runs/teacher_t2a_seed0 --smoke --save-dataset --mujoco-gl disable --out mjx_runs/student_t8_phase_bc_no_previous_action_smoke_seed0
```

正式复现 T8：

```bash
python -m scripts.distill_phase_student_no_previous_action --teacher-run mjx_runs/teacher_t2a_seed0 --save-dataset --strict-acceptance --mujoco-gl disable --out mjx_runs/student_t8_phase_bc_no_previous_action_seed0
```

复查 T8 闭环反馈：

```bash
python -m scripts.diagnose_phase_student_feedback --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t8_phase_bc_no_previous_action_seed0
```

复现失败 T5 的完整根因 audit：

```bash
python -m scripts.audit_phase_student_failure --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t5_phase_bc_seed0 --mujoco-gl disable
```

旧的 phase-free BC/DAgger、`scripts.mjx_train_walk`、`teacher_blend` 退火和 `0.15 m/s`
命令只保留作历史实验或诊断，不是当前部署候选入口。

## 云端提示

- H200 节点可在 EGL 可用时使用 `egl`。
- 当前 RTX 4090 节点缺少可用 EGL 时使用 `--mujoco-gl disable`；这些训练和无渲染诊断不需要 GL context。
- 仅在 OSMesa 配置完整的机器上使用 `--mujoco-gl osmesa`。
- 长训练前先运行单元测试和短 MJX 冒烟，并使用新的输出目录保存配置和评估结果。
