# Extreme Disk Quadruped TODO

> 行走训练任务以 [omnidirectional_training_pipeline.md](omnidirectional_training_pipeline.md) 为准。

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
- [x] actor 观测限制为实机可获得量，历史长度设为 4 帧。

## P0：环境正确性

- [x] 统一 CPU MuJoCo 与 MJX 的逐项奖励和默认参数。
- [x] 移除四足持续接触正奖励、固定世界方向奖励和过强 action-delta 惩罚。
- [x] 增加奖励偏好、command 采样和 action 映射测试。
- [ ] 增加 CPU/MJX 数值逐步对齐测试与单关节方向测试。
- [x] 记录 reward breakdown、速度误差、跌倒率、滑移与 action RMS。
- [ ] 增加 actuator force 和机械功指标。

## P1：示范初始化

- [ ] 用现有 gait/轨迹工具导出 `(observation, command, residual target)` 数据集。
- [ ] 保证数据不要求最终 actor 输入 gait phase。
- [ ] 增加 BC 训练、验证和闭环 rollout 入口。
- [ ] 验证 BC 模型在 teacher 关闭后仍可持续推进。

## P1：PPO 与 Command Curriculum

- [ ] 从 BC checkpoint 启动基础前进 PPO。
- [ ] 在同一 run 内逐步扩展 `vx`，不重置 optimizer。
- [ ] 加入停止、后退、`vy` 和 `wz`，最后联合采样。
- [ ] 固定 evaluation seeds 与 command 网格，按门槛扩大任务空间。
- [ ] 将 imitation loss 平滑衰减到 0，执行无 teacher 验收。

## P2：Sim-to-real

- [ ] 根据 CAD、称重和电机数据更新质量、质心、惯量、增益和力矩范围。
- [ ] 分阶段加入摩擦、动力学、噪声、延迟、地形和外力随机化。
- [ ] 建立训练关节命令到 Pupper position action 的一致性检查。
- [ ] 实机先做悬空、小幅动作、急停和关节限位验证，再落地测试。

## 后续技能

- [ ] 单独训练或优化 `roll_policy`。
- [ ] 将折叠、蹬地和收腿动作改为反馈式 `transition_policy`。
- [ ] 用 FSM 管理 `walk`、`transition` 和 `roll`，最后再评估统一策略或蒸馏。
