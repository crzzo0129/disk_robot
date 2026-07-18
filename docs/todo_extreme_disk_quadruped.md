# Extreme Disk Quadruped TODO

> 当前执行以 [forward_teacher_student.md](forward_teacher_student.md) 为准；[omnidirectional_training_pipeline.md](omnidirectional_training_pipeline.md) 只记录长期路线。

## 已完成基础

- 极端圆盘机身 MJCF 和 12 关节 position actuator 基线。
- `stand`、`folded` keyframe 及姿态诊断、插值和物理仿真脚本。
- Pupper 可视化模型、腿长与碰撞体的初步适配。
- home、folded、后腿蹬地、收腿滚动的开环动作原型。
- CPU 行走 smoke 环境和 MJX PPO 训练入口。
- W&B 指标与训练视频基础链路。

这些完成项表示工具链可运行，不表示旧奖励或 open-loop residual 接口满足最终目标。

## P0：无运行时 IK 的行走契约

- [x] 固化关节顺序、正方向、零位、`q_stand` 与每关节 action scale。
- [x] 将动作目标统一为 `a=tanh(policy_logits)`、`q_target=q_stand + action_scale * a`。
- [x] 移除最终 actor 对 runtime IK、teacher target、预设/实际接触和 privileged observation 的依赖。
- [x] 使用 controller-owned `sin/cos phase + gait_blend` 消除固定周期动作的标签歧义。
- [x] command 统一为机身坐标系 `[vx, vy, wz]`。
- [x] T8 actor 使用四帧 IMU、关节、command 和机身线速度估计，并结构性删除 previous action。
- [x] 用 paired rollout、group counterfactual、Jacobian gain 和单变量消融确认 previous-action 自反馈根因。
- [ ] 单独比较 4/2/1 帧 sensor history；不要把 previous-action 删除结论外推为删除全部历史。
- [ ] 确认 Pupper 实机线速度估计质量；必要时用噪声训练或蒸馏降低依赖。

## P0：环境正确性

- [x] 统一 CPU MuJoCo 与 MJX 的逐项奖励和默认参数。
- [x] 移除四足持续接触正奖励、固定世界方向奖励和过强 action-delta 惩罚。
- [x] 增加奖励偏好、command 采样和 action 映射测试。
- [ ] 增加 CPU/MJX 数值逐步对齐测试与单关节方向测试。
- [x] 记录 reward breakdown、速度误差、跌倒率、滑移与 action RMS。
- [ ] 增加 actuator force 和机械功指标。

## P1：当前 Teacher-Student 基线

- [x] 实现对称 IK v2、privileged residual PPO Teacher、BC 和 DAgger 完整入口。
- [x] 实现不依赖接触传感器的 controller-owned phase clock。
- [x] 增加 IK baseline、最佳 Teacher 保存、checkpoint 恢复和严格验收。
- [x] 使用世界前向净位移阻止通过躯干摇晃刷速度。
- [x] 完成结构参数扫描并生成候选 XML。
- [x] 在云端候选 XML 上运行 IK baseline，并完成 T1b 名义保持。
- [x] 训练并验收 T2a privileged disturbance Teacher（step `1,024,000`）。
- [x] 完成 T3--T7 Student 失败审计，排除 Teacher、phase、标签转换和 actuator 接口。
- [x] 完成 T8 no-previous-action BC 与 Student-only 名义/扰动验收。
- [ ] 本地人工复查 accepted gait 的自碰撞、足端轨迹和部署视频。

## P1：Command-conditioned 手柄控制

- [ ] 冻结 accepted T2a/T8，不把 joystick 直接连接到固定 `vx=0.08` 策略。
- [ ] 设计并实现真正随机化的一维 `vx` Teacher/数据收集，先覆盖停止、启动和低速前进。
- [ ] 将当前 command 从四帧历史的重复项改为单个当前 command，并做单变量兼容实验。
- [ ] 保持 T8 no-previous-action Student observation 契约。
- [ ] 固定 evaluation seeds 与 `vx` 网格，检查稳态误差、切换过渡、扰动恢复和 command counterfactual。
- [ ] 一维 `vx` 通过后加入 `wz`，最后加入 `vy` 与后退。
- [ ] 根据 command-conditioned 结果再决定是否需要 PPO fine-tuning、锚点退火或 phase 自适应。

## P2：Sim-to-real

- [ ] 根据 CAD、称重和电机数据更新质量、质心、惯量、增益和力矩范围。
- [ ] 验证或实现实机 body linear velocity 状态估计；否则从 Student observation 移除。
- [ ] 分阶段加入摩擦、动力学、噪声、延迟、地形和外力随机化。
- [ ] 建立训练关节命令到 Pupper position action 的一致性检查。
- [ ] 实机先做悬空、小幅动作、急停和关节限位验证，再落地测试。

## 后续技能

- [ ] 单独训练或优化 `roll_policy`。
- [ ] 将折叠、蹬地和收腿动作改为反馈式 `transition_policy`。
- [ ] 用 FSM 管理 `walk`、`transition` 和 `roll`，最后再评估统一策略或蒸馏。
