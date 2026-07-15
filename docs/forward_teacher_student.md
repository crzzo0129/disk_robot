# Forward Teacher-Student Pipeline

> 状态：当前可执行训练规范。Teacher、BC、DAgger 和 Student 评估链路已经实现并通过 smoke test，但尚未得到通过正式验收的 Teacher/Student。
>
> 更新日期：2026-07-15

## 1. 当前结论

当前实现和最初方案的主干相同：

1. 从 XML 的 `stand` keyframe 构造对称 trot IK reference。
2. 用 privileged residual PPO 训练 Teacher。
3. Teacher 生成 `(student observation, full position action)` 数据集。
4. Student 先做 Behavior Cloning，再做 DAgger。
5. 最终只部署 Student，不在运行时计算 IK，也不给 Student privileged observation。

但它已经不是最初文档里的原样版本。现在增加了最佳 Teacher 选择、IK baseline、启动混合、世界前向净位移指标、更严格的 Teacher/Student 验收，以及结构候选验证。当前版本是固定 `0.08 m/s`、由命令速度标定 trot reference 的第一阶段，不代表已经解决随机速度、任意方向控制。

正式长训练暂缓。先确认结构候选和 IK reference 能产生真实净前进，再投入 5M Teacher PPO。

## 2. 模型与 stand

默认训练模型：

```text
assets/pupper_v3_disk_structure_candidate.xml
```

原始未缩放几何参考：

```text
assets/pupper_v3_disk_visual.xml
```

候选相对当前模型采用 `hip y = +/-0.09 m`、腿长比例 `0.85`、圆盘半径 `0.20 m`。选择依据见 [structure_variant_study.md](structure_variant_study.md)。在本地固定 trot 扫描中，候选的世界前向速度约为 `0.0277 m/s`，当前结构约为 `0.0201 m/s`；这只是结构筛选结果，不等于云端 Teacher baseline。

`stand` 指 XML 中完整的 keyframe 姿态。即使 12 个关节坐标都是零，各级 body 的四元数、关节轴和局部坐标仍会形成带角度的站姿，因此不能把 `stand` 理解为“十二条关节轴全部处于几何直腿”。IK 中立足端和 Student 的动作基准都从该 keyframe 解析。

## 3. 当前动作定义

Teacher 输出 IK 周围的小残差：

```text
q_target = q_ik(phase) + residual_scale * tanh(teacher_policy)
```

每条腿的 `residual_scale` 为：

```text
[0.10, 0.16, 0.20] rad
```

Student 不读取 IK，直接输出相对 `stand` 的完整位置动作：

```text
q_target = q_stand + student_action_scale * tanh(student_policy)
```

每条腿的 `student_action_scale` 为：

```text
[0.35, 0.60, 0.85] rad
```

默认位置环为 `Kp=10.0`、`Kd=0.4`、力矩上限 `3 Nm`。Teacher 和 DAgger 从 `stand` 重置，并用 25 个控制步，约 0.5 秒，平滑混合到 IK 周期。

## 4. 观测契约

Student 每帧观测为 48 维，堆叠 4 帧后共 192 维：

```text
body angular velocity          3
projected gravity              3
body linear velocity           3
joint position - q_stand      12
joint velocity                12
previous action               12
command [vx, vy, wz]           3
```

当前 `body linear velocity` 来自仿真真值。它没有进入这里所说的 privileged 尾部，但实机仍需状态估计器提供等价量，否则必须在后续训练中加噪、蒸馏掉或移除。现在生成的 Student 不能在没有速度估计的情况下直接上实机。

Teacher 观测为 Student 的 192 维历史加 35 维 privileged 信息，共 227 维：

```text
phase sin/cos                  2
startup blend                 1
desired contacts              4
actual contacts               4
IK tracking error            12
previous residual            12
```

世界坐标前向速度和净位移用于 reward、评估和防止“前后摇晃刷速度”，不输入 Student actor。机身坐标速度仍作为诊断量输出。

## 5. 当前 IK reference

默认参数由 `command_vx` 通过候选结构的实测标定生成：

```text
mode=trot
frequency=1.2 Hz
stride=0.0742 m (command_vx=0.08 时)
height=0.025 m
duty=0.72
command_vx=0.08 m/s
```

候选结构在本地 8 秒物理仿真中，该档实际平均世界前向速度约为 `0.082 m/s`。云端仍必须重新跑 `stage=ik_baseline`，不能借用本地结果代替。使用 `--ik-speed-mode manual` 可恢复手工 `--ik-frequency/--ik-stride` 实验，但正式第一阶段使用命令标定模式。

## 6. 训练阶段

### Stage A1：结构和 IK 闸门

先在本地 Viewer 检查候选结构的站立、摆腿、离地高度、关节方向和是否发生自碰撞：

```powershell
python3.12 scripts\view_ik_gait.py --xml assets\pupper_v3_disk_structure_candidate.xml --training-reference --neutral-pose model --duration 0
```

然后在云端运行候选 smoke：

```bash
python -m scripts.train_forward_teacher_student --smoke --out mjx_runs/forward_008_smoke
```

Smoke 只验证编译、PPO、rollout、BC、DAgger、保存和评估链路。Smoke Student 不通过验收是正常的；需要重点检查 `stage=ik_baseline` 的世界净前进、姿态晃动和失败率。

### Stage A2：Privileged residual PPO Teacher

只有 IK/结构闸门通过后才运行正式训练：

```bash
python -m scripts.train_forward_teacher_student --out mjx_runs/forward_008_v1 --teacher-evals 21 --strict-acceptance
```

正式默认规模为：Teacher 5M environment steps、2048 个并行环境、131072 条示范、Student BC 20000 次更新、2 轮 DAgger、每轮 65536 条 on-policy 状态和 5000 次更新，最终使用 256 个 Student-only 环境评估。

PPO 训练期间会持续评估并保存评分最高的 Teacher。最终验收和数据蒸馏使用最佳参数，不直接使用最后一次更新的参数。

### Stage A3：BC、DAgger 和 Student-only 验收

BC 在 Teacher 状态分布上学习完整位置动作。DAgger 让 Student 闭环运行，再由 Teacher 标注 Student 实际访问到的状态，以减轻分布偏移。最终评估使用 `student` 环境角色，不构建 IK reference，也不计算 phase、期望接触或 privileged observation。

### Stage B：解除固定 gait 锚点

这一阶段尚未实现。为了扩展到 `0.1 m/s` 和更广 command 空间，下一版应先把 IK 变为 command-conditioned reference，再逐步衰减锚点：

```text
q_target = q_stand
         + beta * (q_ik(command, phase) - q_stand)
         + action_scale(beta) * policy_action
```

训练时将 `beta` 从 1 逐步降到 0，同时扩大 `[vx, vy, wz]` 的采样范围。这样 IK 只负责早期探索，不永久限定最终步态。Teacher 通过后再重新蒸馏 Student，并考虑 Student PPO fine-tuning。先完成固定 `0.08 m/s` Student 验收，再把速度命令改为 episode 级采样。

## 7. 验收标准

当前 Teacher 门槛：

- `mean_velocity_x >= 0.06 m/s`
- `failure_rate <= 0.10`
- `mean_velocity_error <= 0.03 m/s`
- `mean_roll_pitch_rate_rms <= 0.50 rad/s`

当前 Student 门槛：

- `mean_velocity_x >= 0.06 m/s`
- `failure_rate <= 0.10`
- `mean_velocity_error <= 0.03 m/s`
- `mean_roll_pitch_rate_rms <= 0.60 rad/s`

这些只是第一阶段低速前进门槛。正式判断还要查看 `mean_forward_distance`、世界速度与 body-frame 速度差异、圆盘触地率、动作饱和和视频。单独达到平均速度不能证明 gait 可用。

`--strict-acceptance` 未通过时保留产物并以退出码 2 结束。Teacher 未通过时不会继续生成示范和训练 Student。

## 8. 输出与评估

```text
mjx_runs/forward_008_v1/
  ik_reference.npz
  teacher/params_best/
  teacher/params_final/
  teacher/ppo_checkpoint/
  teacher/ik_baseline_evaluation.json
  teacher/evaluation.json
  student_policy.npz
  student_policy.json
  evaluation.json
  run_config.json
```

`student_policy.npz` 才是最终部署候选。单独评估：

```bash
python -m scripts.evaluate_forward_student mjx_runs/forward_008_v1/student_policy.npz
```

评估报告必须显示：

```json
{
  "ik_runtime_enabled": false,
  "teacher_observation_enabled": false
}
```

## 9. 恢复训练

```bash
python -m scripts.train_forward_teacher_student --teacher-restore mjx_runs/forward_008_v1/teacher/ppo_checkpoint --out mjx_runs/forward_008_v1_resumed
```

只有 XML、关节顺序、观测、动作定义、reward、控制增益和网络结构完全一致时才允许恢复 checkpoint。任何一项发生变化都应新建 run，不能把旧 checkpoint 当成可兼容初始化。Brax 是否恢复 optimizer state 取决于云端安装版本。
