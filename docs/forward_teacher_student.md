# Forward Teacher-Student Pipeline

## 目标

该入口以 `assets/pupper_v3_disk_visual.xml` 的 `stand` keyframe 为唯一中立姿态，自动完成：

1. 根据 XML 几何生成对称 trot IK v2 周期表。
2. 训练读取 phase、接触状态和 IK 跟踪误差的 privileged residual PPO teacher。
3. 让 teacher 生成 `(student observation, full position action)` 示范。
4. 训练不读取 phase、IK 或接触真值的 student。
5. 让 student 闭环运行并执行 DAgger 修正。
6. 在完全关闭运行时 IK 后评估 student 并输出策略文件。

Teacher 动作：

```text
q_target = q_ik(phase) + residual_scale * teacher_action
```

Student 动作：

```text
q_target = q_stand + student_action_scale * student_action
```

最终部署对象是 student，不是 teacher。Teacher 只用于训练和 DAgger 标注。

这里的 `stand` 是完整 XML 模型姿态，不等同于“十二个关节数值为零时腿是直的”。即使
`stand_q` 的十二个关节坐标都是 `0`，各级 body 的四元数、关节轴和局部坐标系仍决定了
实际带角度的站姿。IK 的中立足端位置和 student 的关节偏移都从这个 keyframe 解析，代码中
不存在另一套 Pupper 默认关节数组。

## 离线 Linux 云端

以下命令默认从 `disk_robot/` 目录运行。云实例无需联网，但 `mjx312` 环境必须已经包含 JAX、MJX、Brax 和 Optax。

先运行完整链路烟测：

```bash
python -m scripts.train_forward_teacher_student --smoke --out mjx_runs/forward_ts_smoke
```

烟测只验证编译、PPO、rollout、BC、DAgger、保存和评估链路，不代表策略已经学会。

正式训练使用一条命令：

```bash
python -m scripts.train_forward_teacher_student --out mjx_runs/forward_ts --strict-acceptance
```

默认正式规模：

- Teacher PPO：5M environment steps，2048 个并行环境。
- Teacher 示范：131072 条。
- Student BC：20000 次更新。
- DAgger：2 轮，每轮 65536 条 student 访问状态和 5000 次更新。
- 最终评估：256 个 student-only 环境，每个 500 control steps。

训练环境分为三种固定角色：`teacher` 使用 IK residual，`dagger` 让 student 执行动作同时让
teacher 生成标签，`student` 只从 `stand` 重置并执行完整关节位置动作。最终评估使用第三种，
不会构建 IK reference，也不会计算 phase、期望接触或 privileged observation。

Teacher 和 DAgger 也从 `stand` 起步，然后默认用 25 个控制步（约 0.5 秒）平滑混合到
对称 IK 周期，避免训练只覆盖“已经处在步态周期中”的状态。混合进度属于 privileged 输入，
不会进入 student observation；可用 `--startup-steps` 调整。

## 输出

成功运行后主要产物位于：

```text
mjx_runs/forward_ts/
  ik_reference.npz
  teacher/params/
  teacher/ppo_checkpoint/
  teacher/evaluation.json
  student_policy.npz
  student_policy.json
  evaluation.json
  run_config.json
```

`student_policy.npz` 是最终策略。其 JSON 伴随文件记录 XML、stand 来源、观测维度、action scale、网络结构和评估结果。

单独重新评估 student：

```bash
python -m scripts.evaluate_forward_student mjx_runs/forward_ts/student_policy.npz
```

评估脚本输出中的以下字段必须为 `false`：

```json
{
  "ik_runtime_enabled": false,
  "teacher_observation_enabled": false
}
```

## 验收

默认门槛：

- student 平均前进速度不低于 `0.04 m/s`；
- 500 control steps 内失败率不高于 `10%`；
- 最终评估环境使用 student 完整位置动作，完全不使用 IK 目标和 teacher residual。

使用 `--strict-acceptance` 时，未达门槛会保留全部产物并以退出码 2 结束，避免把“脚本运行成功”误认为“student 已经学会”。
Teacher 会在生成示范前先执行同样的前进速度和失败率验收；teacher 未达标时不会继续蒸馏。

## 恢复 Teacher

若云实例中断，可恢复 Brax teacher checkpoint：

```bash
python -m scripts.train_forward_teacher_student --teacher-restore mjx_runs/forward_ts/teacher/ppo_checkpoint --out mjx_runs/forward_ts_resumed
```

当前 Brax checkpoint 恢复 policy、value 和 observation normalizer；是否恢复 optimizer state 取决于云端安装的 Brax 版本。
