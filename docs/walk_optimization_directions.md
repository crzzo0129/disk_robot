# Disk Robot 行走训练优化方向

本文记录当前 `disk_robot` 行走训练中观察到的问题、已经做过的调整，以及后续可以尝试的优化方向。重点目标是让机器人从“稳定站住/蹭地”逐步过渡到“稳定向前移动”。

## 当前现象

多次训练和 final policy 视频中观察到：

- 机器人高度通常能维持在目标高度附近，说明基本站立稳定性已经有所改善。
- 行为上常见“先向前蹭一段，然后重心不稳向后倒”。
- `avg_forward_velocity` 可能为负，原因是它统计的是整个 episode 的平均净前向速度；如果后倒时向后位移超过前蹭位移，最终平均速度就是负的。
- 视频观感上腿部活动范围偏小，像“腿被绑住”。
- 当前 `stand` keyframe 前后对称，并且利于向 `folded` 过渡，但可能不是最有利于行走学习的站姿。

## 已经确认/调整过的点

### 1. 初始站姿高度

`stand` keyframe 的 torso 高度已经做过测试，误差约 `0.02mm`，因此当前不把初始高度误差作为主要问题。

### 2. 训练 XML 的物理参数

训练使用 `assets/disk_quadruped_extreme_train.xml`，采用更轻量的参数：

```xml
<option timestep="0.004" integrator="Euler" cone="pyramidal">
```

这是为了 MJX 编译和训练速度。虽然它比本地高精度 XML 更粗糙，但当前不优先回退到更重的物理参数。

### 3. 执行器力限制

训练 XML 已经给 position actuator 加了 `forcerange`，例如：

```xml
forcerange="-8 8"
```

这有助于避免 policy 初期输出导致关节被过强力矩猛拉。

### 4. 稳定化 reward/config

已经尝试过降低速度目标、降低 action 幅度、关闭 feet air time、加入 contact reward、收紧 termination 等方式。后续发现如果 reward 过于保守，策略容易学成“站住不动”；如果 `reward_forward` 太强，则可能变成“乱蹬/向前蹭后后倒”。

## 优化方向一：单独验证 action range 是否过小

当前强烈怀疑“腿像被绑住”的一部分原因来自 action 幅度偏小。

建议先做一个干净实验：

- 不改 stand；
- 不改 XML；
- 不改 reward；
- 只扩大 `action_scale`。

推荐尝试：

```python
action_scale = 0.20
```

如果仍然像绑腿，再试：

```python
action_scale = 0.25
```

判断标准：

- 如果腿部摆动明显增加，说明之前主要受动作范围限制。
- 如果机器人开始乱甩/抽搐，说明 action 放大过头或 force limit/平滑惩罚不足。
- 如果放大后仍然只会蹭地，说明问题更可能来自 stand 腿型或 gait/reward 设计。

当前建议保留：

```python
penalty_action_delta = 0.5
penalty_ang_vel_xy = 0.3
forcerange = "-8 8"
```

这样可以在放大动作范围的同时压制高频抖动。

## 优化方向二：新增行走专用 `walk_stand` keyframe

当前 `stand` keyframe 设计目标是：

1. 前后对称；
2. 向 `folded` 过渡顺畅。

这对折叠/展开是合理的，但它未必适合行走学习。当前前后腿是镜像折叠：

```text
前腿：hip_flex = -0.45, knee =  0.85
后腿：hip_flex =  0.45, knee = -0.85
```

这种姿态会让前后腿关节符号相反，policy 需要学习更复杂的镜像动力学。它可能更容易产生前后对称蹭地，而不是形成明确的推进步态。

建议不要直接破坏原有 `stand/folded` 设计，而是新增一个训练专用 keyframe：

```xml
<key name="walk_stand" ... />
```

然后 MJX reset 使用 `walk_stand`，而 folded 相关逻辑继续使用原来的 `stand` 或 `folded`。

`walk_stand` 的目标：

- 更适合支撑和行走，而不是优先折叠；
- 脚的前后支撑多边形略大；
- 腿在 neutral 附近有足够摆动余量；
- 仍然保证 reset 后无明显穿透和接触冲击。

可以先尝试较温和的腿型，例如：

```text
前腿: hip_flex = -0.35 ~ -0.40, knee =  0.75
后腿: hip_flex =  0.35 ~  0.40, knee = -0.75
```

具体数值需要通过 MuJoCo 检查脚底高度、torso 高度和接触状态。

## 优化方向三：给 observation 加 torso height

当前 observation 包含：

- torso quaternion；
- linear/angular velocity；
- joint position/velocity；
- previous action；
- foot contacts；
- command velocity。

但没有显式 torso height。可是 reward 和 termination 都强依赖：

- `torso_height`；
- `min_torso_height`；
- `target_torso_height`。

这会让 policy 只能间接推断高度，增加学习难度。建议把 torso height 加入 observation。

需要修改：

### `disk_robot/walk_config.py`

当前：

```python
return 4 + 3 + 3 + self.action_size + self.action_size + self.action_size + 4 + 1
```

建议改为加 1：

```python
return 4 + 3 + 3 + 1 + self.action_size + self.action_size + self.action_size + 4 + 1
```

### `disk_robot_mjx/brax_env.py`

在 `_obs()` 中加入：

```python
jp.array([data.xpos[self.torso_body_id][2]]),
```

建议放在速度信息之后、关节信息之前，便于保持语义清晰。

如果仍保留本地 smoke env，也应同步修改 `disk_robot/walk_env.py` 的 `_obs()`，避免 observation size 不一致。

## 优化方向四：加入低权重左右对称奖励

可以考虑加入动作左右镜像对称的小惩罚，用于减少左右乱摆和身体侧倾。

不建议一开始加入前后对称奖励，因为行走需要打破前后对称：后腿推地、前腿摆动/支撑。如果强化前后对称，可能进一步鼓励原地蹭地或前后抵消。

推荐先做左右镜像动作惩罚，权重很小，例如：

```python
penalty_lr_action_symmetry = 0.02
```

动作顺序：

```python
JOINT_NAMES = (
    "fl_hip_abd", "fl_hip_flex", "fl_knee",
    "fr_hip_abd", "fr_hip_flex", "fr_knee",
    "hl_hip_abd", "hl_hip_flex", "hl_knee",
    "hr_hip_abd", "hr_hip_flex", "hr_knee",
)
```

左右镜像误差可以定义为：

```python
front_symmetry_error = (
    (action[0] + action[3]) ** 2
    + (action[1] - action[4]) ** 2
    + (action[2] - action[5]) ** 2
)

hind_symmetry_error = (
    (action[6] + action[9]) ** 2
    + (action[7] - action[10]) ** 2
    + (action[8] - action[11]) ** 2
)

r_lr_symmetry = -penalty_lr_action_symmetry * (front_symmetry_error + hind_symmetry_error)
```

注意：

- 权重应从 `0.01 ~ 0.02` 开始；
- 不建议超过 `0.05`；
- 它只是稳定辅助，不应该主导步态；
- “带时间相位的对称”更接近真实步态，但需要历史动作 buffer 或 gait phase，复杂度更高，建议后续再做。

## 优化方向五：引入 gait phase / 步态先验

如果纯 PPO 继续难以学出自然步态，可以引入轻量 gait phase。比如 trot：

```text
FL + HR 一组
FR + HL 一组
两组相差半个周期
```

需要新增：

- phase state；
- phase 加入 observation；
- stance/swing 接触奖励；
- 可选的 foot clearance 或 target contact schedule。

优点：

- 明确告诉 policy 什么时候该抬脚/落脚；
- 比单纯 feet_air_time 更稳定；
- 更容易形成周期性步态。

缺点：

- 环境复杂度增加；
- phase 频率、占空比需要调；
- 过强的 gait prior 可能限制策略自发发现更适合当前形态的动作。

建议在 action range、obs height、walk_stand 等基础问题处理后再考虑。

## 优化方向六：reward 继续调参的原则

当前不要再盲目增大 `reward_forward`。之前 `reward_forward = 2.0` 时，虽然理论上更鼓励前进，但训练结果显示：

- `avg_forward_velocity` 仍为负；
- `reward_action_delta` 变差；
- `reward_ang_vel_xy` 变差；
- episode 提前终止。

这说明问题不是简单的“前进奖励不够”，而是策略可能通过抖动/蹭地尝试前进，最终失稳。

当前更推荐的 reward 思路：

```python
command_velocity = 0.2
reward_velocity = 1.0
reward_forward = 1.0
tracking_sigma = 0.08
penalty_action_delta = 0.5
penalty_ang_vel_xy = 0.3
reward_contact = 0.05
reward_feet_air_time = 0.0
```

如果机器人稳定但不动：

- 小幅增加 `action_scale`；
- 或略增 `reward_forward` 到 `1.2`；
- 不建议直接大幅提高到 `2.0+`。

如果机器人乱蹬/后倒：

- 增加 `penalty_action_delta`；
- 增加 `penalty_ang_vel_xy`；
- 降低 `action_scale` 或 force limit；
- 不要继续加前进奖励。

## 建议实验顺序

为了保持实验可解释性，建议每次只改一个主要因素。

1. **只扩大 `action_scale`**
   - 试 `0.20` 或 `0.25`。
   - 目的：验证“腿被绑住”是否来自动作幅度不足。

2. **加入 torso height 到 obs**
   - 目的：让 policy 直接知道身体高度，提高防倒能力。

3. **新增 `walk_stand` keyframe**
   - 目的：把折叠友好姿态和行走友好姿态分开。

4. **加入小权重左右对称奖励**
   - 目的：减少左右乱摆，不强制前后对称。

5. **如果仍然难以形成步态，再考虑 gait phase**
   - 目的：用步态先验引导周期性接触。

## 关键观察指标

每次训练后重点看：

- `eval/avg_forward_velocity`
  - 是否从负数接近 0 或变正。

- `eval/avg_torso_height`
  - 是否保持在 `0.38 ~ 0.42`。

- `eval/avg_episode_length`
  - 是否接近 `250`，如果明显小于 250，说明经常提前失败。

- `eval/episode_reward_action_delta`
  - 过负说明动作抖。

- `eval/episode_reward_ang_vel_xy`
  - 过负说明身体 roll/pitch 摇晃严重。

- `eval/episode_reward_forward`
  - 正数表示累计向前，负数表示累计后退。

- `eval/episode_disk_contact_count`
  - 理想接近 0。

- final policy 视频
  - 判断是否是真正迈步，而不是前蹭、滑动或向前扑倒。

## 当前短期建议

下一步最干净的实验是：

```python
action_scale = 0.20 或 0.25
```

其他保持不变，跑一版训练并观察 final 视频。如果腿部活动范围明显改善，再继续做 obs height 或 walk_stand。如果仍然像蹭地，则优先考虑新增 `walk_stand`，而不是继续微调 reward。
