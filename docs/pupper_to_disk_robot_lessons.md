# Pupper 到 Disk Robot 的行走训练迁移对比

本文对比 `pupperv3_mjx` 与当前 `disk_robot` 的 MJX/Brax 行走训练设计，整理可迁移的结构、暂不建议迁移的内容，以及后续实现优先级。

## 背景结论

当前 `disk_robot` 训练中出现的问题包括：

- 放大 `action_scale` 后，机器人更敢动，但平均前向速度仍可能为负。
- final policy 视频常见“先向前蹭，再重心失稳向后倒”。
- 只增大 `reward_forward` 会降低稳定性，导致动作抖动、身体晃动和提前终止。
- 纯靠 velocity reward 让 PPO 自己发现可用步态比较困难。

与 Pupper 相比，`disk_robot` 缺少几个成熟 locomotion scaffold：

1. 行走友好的 default pose / reset pose。
2. observation history。
3. body-frame velocity / gravity-style 姿态观测。
4. contact event、first contact、feet air time 的系统使用。
5. foot slip 惩罚。
6. action latency / IMU latency / noise 等鲁棒性设计。
7. gait phase 或 contact schedule 先验。

因此，后续不应只继续微调 `reward_forward` 或 `action_scale`，而应逐步迁移 Pupper 的训练结构。

---

## 1. Action 设计：default pose + residual action

### Pupper 设计

Pupper walking env 使用 residual action：

```python
motor_targets = self._default_pose + lagged_action * self._action_scale
motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
```

对应代码：

- `pupperv3_mjx/pupperv3_mjx/environment.py:358-366`
- `pupperv3_mjx/pupperv3_mjx/jump_env.py:207-213`

Pupper 还在初始化时覆盖 MJCF 里的 `home` keyframe，使其变成训练用 default pose：

```python
sys.mj_model.keyframe("home").qpos[7:] = default_pose
self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
```

对应代码：

- `pupperv3_mjx/pupperv3_mjx/environment.py:176-193`
- `pupperv3_mjx/pupperv3_mjx/jump_env.py:83-100`

### Disk Robot 当前状态

`disk_robot` 的 MJX env 也使用类似结构：

```python
target_ctrl = neutral_ctrl + config.action_scale * action
```

但核心差别是：

- Pupper 的 `default_pose` 是 locomotion 友好的；
- Disk Robot 原来的 `stand` 更偏 folded/deploy 友好；
- 因此 Disk Robot 新增 `walk_stand` 是合理方向。

### 迁移建议

短期保留：

```text
policy action -> walk_stand neutral_ctrl + action_scale * action
```

后续可以把 `walk_stand` 从 XML keyframe 概念进一步提升为显式 `default_pose`，类似 Pupper：

```python
self.default_pose = jp.array([...])
neutral_ctrl = self.default_pose
```

这样训练姿态与 folded/deploy 姿态可以完全解耦。

---

## 2. Observation 设计：Pupper 用历史观测，Disk Robot 目前是单帧

### Pupper walking obs

Pupper walking env 的 observation 每帧包括：

```python
obs = jp.concatenate([
    lagged_imu_data,                     # angular velocity + gravity
    state_info["command"],               # velocity command
    state_info["desired_world_z_in_body_frame"],
    pipeline_state.q[7:] - self._default_pose + motor_ang_noise,
    state_info["last_act"] + last_action_noise,
])
```

并且堆叠历史：

```python
new_obs_history = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)
```

对应代码：

- `pupperv3_mjx/pupperv3_mjx/environment.py:225-227`
- `pupperv3_mjx/pupperv3_mjx/environment.py:485-543`

Pupper jump env 也类似，观测包含：

- IMU；
- height command；
- joint offset；
- last action；
- observation history。

对应代码：

- `pupperv3_mjx/pupperv3_mjx/jump_env.py:124-126`
- `pupperv3_mjx/pupperv3_mjx/jump_env.py:305-354`

### Disk Robot 当前状态

Disk Robot 当前 observation 已经包含：

- quaternion；
- linear velocity；
- angular velocity；
- torso height；
- joint position；
- joint velocity；
- previous action；
- foot contacts；
- command velocity。

但仍是**单帧观测**。

### 为什么 history 重要

行走不是静态控制问题。单帧很难判断：

- 当前脚是刚离地还是快落地；
- 身体是在向前恢复还是正在向后倒；
- 当前动作处在步态周期哪一段；
- 过去几步 action 是否在振荡。

Pupper 使用 observation history 是它训练稳定的重要原因之一。

### 迁移建议

优先级：高。

建议给 Disk Robot 增加：

```python
observation_history: int = 4
```

先从 4 帧开始，不必直接上 Pupper 的 20 帧。实现方式：

- `info` 中保存 `obs_history`；
- `_obs()` 先构造单帧 `obs_frame`；
- reset 时初始化 `obs_history = zeros(frame_size * history)`；
- step 时 roll 并插入最新帧。

注意：Brax PPO 网络输入维度会变大，旧模型参数不能复用。

---

## 3. 坐标系：Pupper 用 body-frame velocity，Disk Robot 目前偏 world-x

### Pupper 设计

Pupper 的线速度 tracking 是在 body frame 中计算：

```python
local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
```

对应代码：

- `pupperv3_mjx/pupperv3_mjx/rewards.py:58-63`

角速度 tracking 也转到 body frame：

```python
base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
```

对应代码：

- `pupperv3_mjx/pupperv3_mjx/rewards.py:66-70`

### Disk Robot 当前状态

Disk Robot 当前 forward velocity 是通过 torso world x 位移计算：

```python
forward_velocity = (x_new - x_old) / dt
```

这在机器人 yaw 不大时可以用，但如果机器人转头、侧滑或旋转，world-x reward 会混淆：

- 朝向不对但沿 world x 滑动，也可能得分；
- 身体旋转时，真正的“向前”与 world x 不一致；
- 对圆盘机器人尤其容易出现 yaw/roll/pitch 耦合。

### 迁移建议

中期建议把 velocity reward 改成 body-frame forward velocity：

```text
body_forward_velocity = dot(world_velocity_xy, torso_forward_axis_xy)
```

在 MJX 中可用 torso rotation matrix 或 quaternion 计算。这样 reward 的语义更接近：

```text
沿身体朝向向前走
```

而不是：

```text
世界坐标 x 方向位移
```

同时可加入 heading/yaw 稳定项，避免圆盘机器人边转边滑。

---

## 4. 接触事件：Pupper 系统使用 contact / last_contact / first_contact

### Pupper 设计

Pupper walking env 对脚接触做了事件滤波：

```python
foot_contact_z = foot_pos[:, 2] - self._foot_radius
contact = foot_contact_z < 1e-3
contact_filt_mm = contact | state.info["last_contact"]
contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
state.info["feet_air_time"] += self.dt
```

step 末尾：

```python
state.info["feet_air_time"] *= ~contact_filt_mm
state.info["last_contact"] = contact
```

对应代码：

- `pupperv3_mjx/pupperv3_mjx/environment.py:374-381`
- `pupperv3_mjx/pupperv3_mjx/environment.py:448-453`
- `pupperv3_mjx/pupperv3_mjx/jump_env.py:221-233`
- `pupperv3_mjx/pupperv3_mjx/jump_env.py:286-290`

### Disk Robot 当前状态

Disk Robot 已经有：

- `foot_contacts`；
- `last_foot_contacts`；
- `feet_air_time`；
- `first_contact`。

但当前 `reward_feet_air_time = 0.0`，接触事件还没有真正发挥引导步态的作用。

### 迁移建议

不要急着把 `feet_air_time` 开很大。建议按顺序：

1. 先记录并观察：
   - `foot_contact_count`；
   - `first_contact_count`；
   - `feet_air_time`。

2. 加 foot slip 惩罚，减少蹭地。

3. 再小幅打开 `feet_air_time`：

```python
reward_feet_air_time = 0.02 ~ 0.05
```

只在机器人能保持 episode length 接近满长后开启。

---

## 5. Foot slip：Disk Robot 当前非常需要迁移

### Pupper 设计

Pupper 对接触中的脚水平速度做惩罚：

```python
return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))
```

对应代码：

- `pupperv3_mjx/pupperv3_mjx/rewards.py:109-124`
- `pupperv3_mjx/pupperv3_mjx/environment.py:431-436`

### 为什么 Disk Robot 需要

Disk Robot 视频现象像：

```text
脚在地上蹭 -> 身体前后滑 -> 重心失稳 -> 后倒
```

如果没有 foot slip penalty，policy 可能把“拖脚蹭地”当成一种尝试移动的方式，而不学习真正的抬脚/落脚。

### 迁移建议

优先级：高。

Disk Robot 可先实现简化版：

- 找到四个 foot geom 的 world position；
- 用相邻 step 的 foot position 差分估计 foot xy velocity；
- 只在 foot contact 时惩罚 xy velocity。

需要在 `info` 中新增：

```python
last_foot_pos
```

reward：

```python
foot_xy_vel = (foot_pos_xy - last_foot_pos_xy) / dt
r_foot_slip = -penalty_foot_slip * sum(contact * ||foot_xy_vel||^2)
```

初始权重建议：

```python
penalty_foot_slip = 0.05 ~ 0.2
```

不要太大，否则机器人可能完全不敢挪脚。

---

## 6. Reward 组织：Pupper 更重视稳定、平滑、碰撞、滑移

### Pupper walking reward scales

Pupper 默认 walking config：

```python
tracking_lin_vel=1.5
tracking_ang_vel=0.8
lin_vel_z=-2.0
ang_vel_xy=-0.05
orientation=-5.0
tracking_orientation=1.0
torques=-0.0002
joint_acceleration=-1e-6
action_rate=-0.01
feet_air_time=0.2
stand_still=-0.5
stand_still_joint_velocity=-0.1
abduction_angle=-0.1
termination=-100.0
foot_slip=-0.1
knee_collision=-1.0
body_collision=-1.0
```

对应代码：

- `pupperv3_mjx/pupperv3_mjx/config.py:14-64`

### Disk Robot 当前 reward 特点

Disk Robot 当前 reward 更直接：

- forward velocity tracking；
- forward linear reward；
- upright；
- height target；
- action/action delta；
- angular velocity；
- disk contact；
- termination。

缺少：

- foot slip；
- torque / actuator work；
- joint acceleration；
- observation history；
- command resampling；
- zero-command stand-still curriculum；
- explicit orientation tracking in body frame。

### 迁移建议

短期不要照抄所有项。优先迁移：

1. `foot_slip`
2. `action_rate` / 当前已有 `action_delta`，继续保留
3. `joint_acceleration`，如果能稳定拿到上一帧 joint velocity
4. `orientation` / `tracking_orientation`
5. `feet_air_time` 小权重

暂缓迁移：

- torque / mechanical work：Disk Robot 当前 actuator/force 数据链路需要先确认；
- heavy domain randomization；
- command y/yaw 全范围随机。

---

## 7. Jump 环境的启发：事件式任务比纯速度任务更容易

### Pupper jump 设计

Pupper jump env 用事件式奖励：

- peak height；
- takeoff velocity；
- time in air；
- landing stability；
- landing impact；
- airborne angular velocity；
- horizontal drift；
- action symmetry。

对应代码：

- `pupperv3_mjx/pupperv3_mjx/jump_env.py:240-284`
- `pupperv3_mjx/pupperv3_mjx/jump_rewards.py:10-82`
- `pupperv3_mjx/pupperv3_mjx/jump_config.py:14-33`

### 对当前问题的启发

你提到 Pupper 从速度跟随改成跳跃没有这么难，这有合理原因：

```text
跳跃可以是四腿同步压缩/伸展，是更同步的全身事件。
行走需要相位、支撑腿、摆动腿、重心转移和落脚时机。
```

对 Disk Robot 来说，纯速度 tracking 难，是因为它没有告诉 policy 如何组织腿的时序。

因此应考虑引入 gait phase 或 contact schedule。

---

## 8. Gait phase / contact schedule：课程思路应作为中期重点

### Pupper 代码现状

当前读到的 `pupperv3_mjx` 代码中，未发现显式 `gait_phase` 状态变量；更多是通过 contact event 和 feet air time 隐式塑造步态。

但你提到课程中有 gait phase 规划，这对 Disk Robot 很有价值。

### 为什么 Disk Robot 更需要 gait phase

Disk Robot 的形态更不常规：

- 圆盘身体；
- 前后支撑距离短；
- stand/folded 对称性强；
- 足端小球接触；
- 腿段不参与碰撞；
- 纯 velocity reward 容易学成蹭地/后倒。

因此它比 Pupper 更需要相位先验。

### 最小实现方案

增加 phase：

```python
phase = (step_count * dt * gait_frequency) % 1.0
phase_obs = [sin(2*pi*phase), cos(2*pi*phase)]
```

加入 observation。

定义 trot contact schedule：

```text
phase in [0, 0.5):
  stance: FL + HR
  swing:  FR + HL

phase in [0.5, 1):
  stance: FR + HL
  swing:  FL + HR
```

低权重 reward：

```python
r_gait_contact = reward_gait_contact * mean(
    stance_mask * foot_contacts + swing_mask * (1 - foot_contacts)
)
```

初始权重建议：

```python
reward_gait_contact = 0.1 ~ 0.3
```

注意：不要强制太死，否则 policy 可能为了满足接触表而牺牲稳定性。

### 后续增强

在 contact schedule 稳定后，再加：

- swing foot clearance；
- stance foot slip penalty；
- phase-dependent feet air time；
- gait frequency curriculum。

---

## 9. Action symmetry：可借鉴但要谨慎

### Pupper jump symmetry

Pupper jump env 有 action symmetry：

```python
return (
    jp.sum(jp.abs(action[0:3] - action[3:6]))
    + jp.sum(jp.abs(action[6:9] - action[9:12]))
    + jp.sum(jp.abs(action[0:6] - action[6:12]))
)
```

对应代码：

- `pupperv3_mjx/pupperv3_mjx/jump_rewards.py:65-71`

这适合跳跃，因为跳跃可以是全身同步动作。

### Disk Robot walking 不宜照抄

行走需要打破前后对称：

```text
后腿推地，前腿摆动/支撑。
```

如果照抄前后对称奖励，可能强化当前问题：

```text
前后动作互相抵消 -> 原地蹭地 -> 不能向前
```

### 推荐迁移方式

只加低权重左右镜像对称，暂不加前后对称：

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
```

权重：

```python
penalty_lr_action_symmetry = 0.01 ~ 0.02
```

优先级低于 gait phase 和 foot slip。

---

## 10. Domain randomization / latency / noise：后期再迁移

### Pupper 设计

Pupper 加入了：

- start position randomization；
- action latency buffer；
- IMU latency buffer；
- angular velocity noise；
- gravity noise；
- motor angle noise；
- last action noise；
- random kicks。

对应代码：

- `pupperv3_mjx/pupperv3_mjx/environment.py:71-120`
- `pupperv3_mjx/pupperv3_mjx/environment.py:300-312`
- `pupperv3_mjx/pupperv3_mjx/environment.py:348-366`
- `pupperv3_mjx/pupperv3_mjx/environment.py:498-523`

### Disk Robot 迁移建议

当前阶段不建议马上加。原因：

- 机器人还没学会稳定向前；
- 过早随机化会增加学习难度；
- 当前目标是先找到能走的 nominal policy。

后期迁移顺序：

1. observation noise；
2. small reset noise；
3. action latency；
4. external push/kick；
5. friction/mass randomization。

---

## 11. 推荐迁移路线图

### Phase 0：已完成/正在做

- 新增 `walk_stand` keyframe。
- MJX reset 使用 `walk_stand`。
- observation 加 torso height。
- `action_scale` 实验验证动作范围。

### Phase 1：最应该马上做

1. 加 observation history，先 `history=4`。
2. 加 foot slip penalty。
3. 将 velocity tracking 改为 body-frame forward velocity。
4. 保留当前 `action_delta` 和 `ang_vel_xy` 平滑/稳定项。

### Phase 2：加入步态先验

1. 加 `phase` 到 info。
2. obs 加 `sin_phase/cos_phase`。
3. 加低权重 trot contact schedule reward。
4. 观察 `first_contact_count`、`foot_contact_count`、视频是否形成周期性。

### Phase 3：优化步态质量

1. 小幅恢复 `feet_air_time`。
2. 加 swing clearance。
3. 加左右镜像 symmetry 小惩罚。
4. 调整 gait frequency。

### Phase 4：鲁棒性迁移

1. sensor/action noise。
2. action latency。
3. reset randomization。
4. push/kick perturbation。
5. sim-to-real 相关 domain randomization。

---

## 12. 最小可执行下一步建议

如果下一步要从 Pupper 迁一个最有价值的功能，建议优先顺序是：

```text
1. foot slip penalty
2. observation history
3. body-frame velocity reward
4. gait phase/contact schedule
```

原因：

- foot slip 直接针对当前“蹭地/滑动”问题；
- observation history 让 policy 能感知动态趋势；
- body-frame velocity 让“前进”的定义更合理；
- gait phase 降低 PPO 自己发明步态的难度。

不要优先迁移：

- Pupper jump 的全身前后 symmetry；
- 大规模 domain randomization；
- 强 feet_air_time；
- 过强 forward reward。

---

## 13. 总结

Pupper 之所以更容易训练，不只是 reward 权重好，而是整个 locomotion 环境有完整结构：

```text
default pose
+ residual action
+ observation history
+ IMU/gravity style orientation
+ contact event tracking
+ feet air time
+ foot slip
+ action rate
+ collision/termination
+ command curriculum/randomization
```

Disk Robot 当前主要还在 reward 和 action_scale 层面调参。要继续推进，应该开始迁移 Pupper 的结构性设计，尤其是：

```text
foot slip + obs history + body-frame velocity + gait phase
```

这比继续单独调 `reward_forward` 更可能解决“前蹭后倒、平均速度为负、腿像被绑住”的问题。
