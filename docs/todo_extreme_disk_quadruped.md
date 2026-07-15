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

## P0：无 Gait 行走契约

- [x] 固化关节顺序、正方向、零位、`q_stand` 与每关节 action scale。
- [x] 将动作目标统一为 `a=tanh(policy_logits)`、`q_target=q_stand + action_scale * a`。
- [x] 移除最终 actor 对 gait phase、teacher target 和预设接触时序的依赖。
- [x] command 统一为机身坐标系 `[vx, vy, wz]`。
- [x] actor 使用 IMU、关节、上一动作、command 和机身线速度估计，历史长度设为 4 帧。
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
- [x] 保证最终 actor 不输入 gait phase。
- [x] 增加 IK baseline、最佳 Teacher 保存、checkpoint 恢复和严格验收。
- [x] 使用世界前向净位移阻止通过躯干摇晃刷速度。
- [x] 完成结构参数扫描并生成候选 XML。
- [ ] 本地人工验收结构候选的 IK gait、自碰撞和足端轨迹。
- [ ] 在云端候选 XML 上重新运行 IK baseline 和 smoke。
- [ ] 训练并验收正式 Teacher；未通过时先改结构、IK 或控制，不继续蒸馏。
- [ ] Teacher 通过后完成 BC、DAgger 和 Student-only 验收。

## P1：解除 Gait 锚点与 Command Curriculum

- [ ] 实现 command-conditioned IK reference。
- [ ] 实现 IK 锚点权重 `beta` 从 1 到 0 的退火，并同步扩大完整动作范围。
- [ ] 从 BC/DAgger Student 启动无 IK 的 PPO fine-tuning。
- [ ] 在同一 run 内逐步扩展 `vx`，不重置 optimizer。
- [ ] 加入停止、后退、`vy` 和 `wz`，最后联合采样。
- [ ] 固定 evaluation seeds 与 command 网格，按门槛扩大任务空间。
- [ ] 在 `0.1 m/s` 前进通过后再进入全向任务。

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
