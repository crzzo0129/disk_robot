# Disk Robot 无 Gait 全向行走训练 Pipeline

> 状态：当前训练唯一依据，Stage 0 与无 gait PPO 环境已实现  
> 更新日期：2026-07-14

## 1. 目标与核心决定

目标是在可接受速度范围内，根据机身坐标系命令 `command = [vx, vy, wz]` 实现任意方向平移和转向。最终策略不依赖运行时开环 gait、固定相位、固定落足顺序或固定速度模板。

策略沿用 Pupper 平台容易部署的位置残差接口：

```text
a        = tanh(policy_logits(observation, command))
q_target = q_stand + action_scale * a
```

- `q_stand` 是静态中立站姿，不随相位变化。
- `action_scale` 是按关节设定的固定向量，不做逐轮放宽 curriculum。
- 策略输出 12 个 `[-1, 1]` 无量纲位置残差。Brax policy distribution 已完成 tanh，环境只做边界裁剪，再由 MuJoCo position actuator 或实机位置环跟踪。
- 开环 gait 只允许作为离线 teacher 生成示范，不能进入最终策略的运行时动作计算。

## 2. 为什么放弃运行时 Gait 锚点

旧环境中，零残差已经能由开环 gait 前进，RL 学到的更像是“不要破坏 gait”。同时奖励偏爱四足持续接触、固定世界方向和小动作，使静止成为接近最优的局部解。

运行时 gait 还会限制步态拓扑和 command 空间，并让机械改型与控制模板强耦合。这里保留“示范带来较好的初始化”，去掉“运行时强锚点”。

## 3. 控制与观测契约

### 3.1 控制频率

- 物理仿真建议维持 `dt = 0.004 s`。
- 每 5 个物理步更新一次策略，即 50 Hz 控制频率。
- position actuator 的 `kp`、阻尼和力矩上限应落在 Pupper 实机可实现范围。
- 训练和部署必须使用同一关节顺序、正方向、零位与 action scale。

### 3.2 Actor 观测

只使用实机可获得或可靠估计的量：

```text
body angular velocity          3
projected gravity              3
estimated body velocity        3
joint position - q_stand      12
joint velocity                12
previous action               12
command [vx, vy, wz]           3
```

建议堆叠最近 3 至 5 帧，而不是旧环境的 20 帧。当前使用 4 帧、每帧 48 维，总计 192 维。

当前 XML 已声明 `global_linvel`，基础学习阶段把机身线速度估计放入 actor，使速度命令成为可观测任务。实机部署必须提供对应状态估计；若实机估计不可靠，应在后续通过观测噪声、速度估计器或 teacher-student 蒸馏移除该依赖，不能直接用不可获得的仿真真值上线。

### 3.3 Command 采样

- command 在 episode 内每 1 至 3 秒重采样，而不是每步跳变。
- 预留 10% 至 20% 的零速度命令，学习稳定站立和停止。
- 初期先训练 `vx > 0`，随后扩展到 `vx` 正负、`vy` 和 `wz`。
- command 和实际速度统一转换到机身坐标系计算奖励。

## 4. 奖励设计

奖励只表达任务目标和必要的物理约束，不指定步态形状：

```text
r_xy  = exp(-||v_body_xy - command_xy||^2 / sigma_v)
r_yaw = exp(-(yaw_rate - command_wz)^2 / sigma_w)
```

起始权重建议为平面速度跟踪 `1.0`、偏航角速度跟踪 `0.5`、零命令稳定站立 `0.1`。精确数值由 reward breakdown 和基准 rollout 调整。

弱惩罚包括：竖直速度、roll/pitch、圆盘机身触地、足端滑移、软限位、力矩/机械功和 action rate。初始阶段不要加入：

- 四只脚同时接触地面的正奖励。
- 固定世界 `+X` 朝向奖励。
- 固定 gait phase、落足顺序或抬腿时刻。
- 过大的 action、action delta 或姿态惩罚。

## 5. 高效训练阶段

### Stage 0：先证明环境可学

- 静止、随机和简单周期策略的 return 可区分。
- 非零 command 下，完美跟踪的理论奖励明显高于静止。
- CPU MuJoCo 与 MJX 对同一状态、动作的奖励逐项一致。
- command 确实进入 observation；action 正负能正确改变关节目标。
- 无 NaN、初始穿透弹飞或关节索引错误。

当前代码中 CPU 与 MJX 的 lateral penalty、air-time 项和部分默认 command 不一致，正式训练前必须统一。

### Stage 1：Teacher 引导与策略模仿

当前已在 `pupper_v3_disk_visual.xml` 上搜索并固化一组 teacher 轨迹。CPU 验证中它连续运行 7 秒、前进约 1.1 m 且没有触发跌倒终止。训练使用动作混合：

```text
teacher_action = normalized_teacher_joint_offset
applied_action = teacher_blend * teacher_action
               + (1 - teacher_blend) * policy_action
```

actor 仍不接收 gait phase。引导阶段通过 `teacher_imitation` 奖励让 policy 从状态历史和上一动作中复现 teacher，再逐步降低 `teacher_blend`。当前 Brax checkpoint 恢复会迁移观测归一化、policy 和 value 参数，但 optimizer 会重新初始化。

建议阶段：

1. `teacher_blend=1.0`、imitation `1.0`：学习 teacher 动作。
2. `teacher_blend=0.5`、imitation `0.3`：策略开始承担真实动力学控制。
3. `teacher_blend=0.0`、imitation `0.0`：纯策略验收。

验收门槛：第三阶段关闭 teacher 后仍能连续前进。teacher 与当前几何绑定；修改腿长、关节方向或站姿后必须重新运行 teacher 回归验证或重新搜索参数。

### Stage 2：基础 PPO

- 从 BC 权重初始化 PPO。
- command 先限制为有足够奖励区分度的前进范围；当前使用 `vx in [0.15, 0.30] m/s`，`vy = wz = 0`，且基础阶段不采样零命令。
- episode 中包含多个非零速度，不只训练单一 `0.1 m/s`。
- 使用连续采样分布和同一训练 run，不因小阶段重置 optimizer。
- 若动作长期贴近零，先修奖励差，而不是放大 action scale。

验收门槛：速度误差下降、跌倒率可控、动作不是零输出，且关闭 teacher 后仍能前进。

### Stage 3：Command-space Curriculum

按任务空间扩展，而不是按 action scale 扩展：

1. 扩大正向速度范围。
2. 加入减速、停止和后退。
3. 加入横向速度 `vy`。
4. 加入偏航角速度 `wz`。
5. 联合采样 `[vx, vy, wz]`，增加快速 command 切换。

仅当最近窗口的速度误差、存活率和跌倒率达标时扩大分布，并持续采样一部分旧 command 防止遗忘。

### Stage 4：移除示范依赖

若 PPO 仍使用 imitation loss，将其平滑衰减到 0。最终验收模型不得读取 teacher action、gait phase 或预设足端轨迹。

### Stage 5：Sim-to-real 随机化

依次加入：地面参数；关节阻尼、armature、增益和力矩上限；质量、质心和惯量；观测噪声、延迟和 action 保持；坡度、地面扰动和外力。随机化范围以真实测量为中心，CAD、称重和电机数据确定后应收窄分布。

## 6. 训练规模与评估

- `10k` steps 只用于编译和指标冒烟，不能判断算法是否有效。
- 基础前进策略至少按百万级 environment steps 评估。
- 完整全向策略预留 20M 至 50M steps，再由学习曲线决定是否继续。
- MJX 可从 1024 至 4096 并行环境起步，以显存和吞吐为准。
- 固定一组 evaluation seeds 和 command 网格，不参与训练采样。

每次评估至少记录 `vx/vy/wz` RMSE、跌倒率、存活时间、圆盘触地率、力矩/机械功、足端滑移、action RMS、action-rate RMS 和 command step response。视频只做定性检查。

## 7. 行走、滚动与转换

先拆分技能，而不是让一个策略同时学习全部模式：

- `walk_policy`：当前 pipeline 的全向行走。
- `roll_policy`：圆盘接地后的滚动控制。
- `transition_policy`：站立、折叠、蹬地和收腿转换。
- 上层 FSM 或 mode policy：根据状态与命令切换技能。

先让每个技能在各自初始状态分布内稳定，再训练转换，最后才评估统一 policy 或蒸馏。

## 8. 近期实施顺序

1. 固化关节顺序、`q_stand`、action scale 和实机控制频率。
2. 将动作从 `open_loop_target + residual` 改为 `q_stand + residual`。
3. 统一 CPU/MJX observation、command、reward 与 termination。
4. 增加 Stage 0 测试和 reward breakdown 日志。
5. 建立 teacher dataset 导出与 BC 训练入口。
6. 完成基础前进 PPO，再扩展 command curriculum。
7. 加入 domain randomization 和 Pupper 平台回放检查。

当前 `walk_smoke.py`、`mjx_train_walk.py` 与 CPU/MJX 环境已经实现静态站姿残差、teacher blend 退火、checkpoint 权重恢复、三维 command 和共享奖励。尚未实现的是 optimizer-state 连续恢复、自动按指标切换阶段与分阶段 domain randomization。
