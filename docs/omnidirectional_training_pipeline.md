# Disk Robot 全向训练长期路线

> 状态：历史路线与长期目标，不是当前代码的运行说明。
>
> 当前可执行入口以 [forward_teacher_student.md](forward_teacher_student.md) 为准。
>
> 更新日期：2026-07-15

## 最终目标

最终策略接收机身坐标系命令 `command = [vx, vy, wz]`，完成可接受速度范围内的任意方向平移和转向。部署时不依赖固定 gait phase、固定落足顺序或运行时 IK reference。

```text
a        = tanh(student_policy(observation, command))
q_target = q_stand + action_scale * a
```

该目标保持不变，但原文中“直接从 stand 训练无 gait PPO”“0.15 至 0.30 m/s 已是当前基础阶段”“Teacher blend 是当前主线”等描述已经过时。从零 PPO 已经证明探索效率不足，当前先采用对称 IK v2、privileged residual PPO Teacher、BC 和 DAgger 建立可行行为，再逐步释放 gait 锚点。

## 当前到长期目标的路线

1. **低速可行性**：在 `0.08 m/s` 固定前进任务上验证结构、速度标定 IK、Teacher 和 Student 蒸馏链路。
2. **Command-conditioned IK**：让步频、步长、抬脚高度等随 `[vx, vy, wz]` 变化，而不是只使用一个固定周期。
3. **锚点退火**：逐步减小 IK reference 权重，同时扩大策略的完整位置动作范围。
4. **Command curriculum**：依次加入更高前进速度、停止、后退、横移和偏航，保留旧 command 分布防止遗忘。
5. **Student fine-tuning**：在无 IK、无 privileged observation 的条件下继续 PPO 或其他闭环优化。
6. **Sim-to-real**：加入电机、延迟、摩擦、质量、质心、传感噪声和外扰随机化，并移除无法在实机可靠估计的观测。
7. **滚动技能**：行走、折叠/蹬地转换和滚动分别训练，再由上层状态机或 mode policy 管理。

## 必须保持的部署约束

- 最终输出为 12 个关节位置目标，关节顺序、正方向和零位必须与 Pupper 实机一致。
- Student 只能使用实机可测量或可可靠估计的状态。
- 世界坐标速度可以作为训练 reward 或 Teacher 的 privileged 判据，不能假定部署时直接获得。
- 当前 Student observation 中仍有仿真 body linear velocity；实机部署前必须提供状态估计器，或通过训练移除该依赖。
- 每次改变 XML、动作、观测、reward、控制增益或 command 定义，都要重新验证 checkpoint 和数据集兼容性。

## 里程碑

| 里程碑 | 通过条件 | 当前状态 |
|---|---|---|
| 结构与 IK baseline | 有真实净前进，晃动和饱和可接受 | 候选待人工可视化和云端复测 |
| Privileged Teacher | 通过速度、失败率、误差和姿态门槛 | 链路完成，正式策略未通过 |
| Gait-free Student | Student-only 闭环通过同类门槛 | 链路完成，正式策略未通过 |
| 0.1 m/s 前进 | command-conditioned reference 与锚点退火后通过 | 未实现 |
| 全向控制 | `[vx, vy, wz]` 网格评估通过 | 未实现 |
| 实机部署 | 状态估计、延迟和电机约束闭环通过 | 未实现 |

旧版本的阶段参数不再作为训练依据。新增实现或调整默认值时，应优先更新 `forward_teacher_student.md`；本文件只记录跨阶段长期方向。
