# Disk Robot 全向训练长期路线

> 状态：历史路线与长期目标，不是当前代码的运行说明。
>
> 当前可执行入口以 [forward_teacher_student.md](forward_teacher_student.md) 为准。
>
> 更新日期：2026-07-18

## 最终目标

最终策略接收机身坐标系命令 `command = [vx, vy, wz]`，完成可接受速度范围内的任意方向平移和转向。部署时不依赖运行时 IK reference、足部接触或 privileged observation。当前已验证的 T8 使用 controller-owned phase clock；全向阶段可以让 command 调节 phase frequency/blend，是否最终移除 phase 必须由后续消融决定，不能把“内部时钟”误等同于外部 gait/接触输入。

```text
a        = tanh(student_policy(observation, command))
q_target = q_stand + action_scale * a
```

该目标保持不变，但原文中“直接从 stand 训练无 gait PPO”“0.15 至 0.30 m/s 已是当前基础阶段”“Teacher blend 是当前主线”等描述已经过时。从零 PPO 已经证明探索效率不足。固定速度阶段已经由对称 IK v2、privileged residual PPO Teacher 和 T8 no-previous-action BC 打通；DAgger 没有解决旧 Student 的高增益自反馈，不是下一阶段的默认步骤。

## 当前到长期目标的路线

1. **低速可行性**：已在 `0.08 m/s` 固定前进任务上验证 T2a Teacher 和 T8 Student。
2. **一维 command conditioning**：先随机化 `vx`，覆盖停止、启动和低速前进；保持 no-previous-action 契约。
3. **Command/phase 联动**：让 controller phase 的 frequency/blend 随命令变化，并验证速度切换而不是只看稳态点。
4. **Command curriculum**：一维通过后加入偏航，再加入后退和横移，保留旧 command 分布防止遗忘。
5. **必要时闭环 fine-tuning**：只有 BC retention 或命令过渡不够时，才引入 Student PPO、锚点退火或其他闭环优化。
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
| 结构与 IK baseline | 有真实净前进，晃动和饱和可接受 | 云端固定速度 baseline 已通过，仍需部署视频复查 |
| Privileged Teacher | 名义保持且扰动恢复改善 | T2a 已通过，step `1,024,000` |
| Fixed-speed Student | 无 IK/contact/previous action 的 Student-only 闭环通过 | T8 已通过，名义/扰动 `vx=0.0823/0.0785` |
| 1-D joystick control | `vx` 网格和停止/启动过渡通过 | 未实现 |
| 全向控制 | `[vx, vy, wz]` 网格评估通过 | 未实现 |
| 实机部署 | 状态估计、延迟和电机约束闭环通过 | 未实现 |

旧版本的阶段参数不再作为训练依据。新增实现或调整默认值时，应优先更新 `forward_teacher_student.md`；本文件只记录跨阶段长期方向。
