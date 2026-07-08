# Disk Robot：Open-loop Gait + Residual RL 实现路线

## 1. 当前判断

当前项目已经满足两个重要前提：

1. **固定 stand 姿态可以站住**
   - 说明机器人结构、初始站姿、PD 控制、接触参数至少基本可行。
   - 不需要优先训练一个纯静态 stand policy。

2. **open-loop gait 已经可以走**
   - `disk_robot/scripts/open_loop_gait.py` 里的周期关节目标已经能产生前进运动。
   - 当前主要问题是：机器人走远后会逐渐歪掉、侧漂或偏航。
   - 这说明核心瓶颈不是“不会走”，而是“缺少闭环反馈修正”。

因此，当前最合适的路线不是从零训练 walking policy，也不是重新训练 stand balance，而是：

```text
open-loop gait + residual RL stabilization
```

也就是：

```text
ctrl = open_loop_gait_target + residual_policy(obs)
```

---

## 2. 总体目标

目标不是让 RL 从零发明步态，而是让 RL 学会对已有 open-loop gait 进行小幅修正。

RL policy 主要负责：

- 抑制 yaw drift；
- 抑制 lateral drift；
- 修正 torso roll/pitch；
- 维持 torso height；
- 修正左右腿推力不平衡；
- 修正脚接触时机误差；
- 在不破坏原有 gait 的情况下延长稳定行走距离。

---

## 3. 控制结构

当前 open-loop 控制逻辑可以抽象成：

```python
gait_target = make_targets(neutral, t, args)
ctrl = gait_target
```

后续改成：

```python
gait_target = make_targets(neutral, t, gait_params)
residual = residual_scale * policy(obs)
ctrl = gait_target + residual
ctrl = clip(ctrl, ctrl_low, ctrl_high)
```

其中：

```text
gait_target:
    来自已有 open-loop gait。

residual:
    RL policy 输出的 12 维关节目标修正量。

residual_scale:
    初期应明显小于 walking 的 action_scale。
```

建议初始：

```text
residual_scale = 0.03 ~ 0.08
```

不要一开始用 `0.3` 这种较大 action scale，否则 RL 会轻易破坏已有 open-loop gait。

---

## 4. 为什么暂时不优先训练 stand balance

在当前条件下，单独训练 stand balance 不是必经阶段。

原因是：

```text
固定 stand 已经通过
open-loop 已经能走
问题主要发生在动态行走过程
```

stand balance 的状态分布主要是：

```text
低速度
固定支撑
固定关节目标附近
无周期性摆腿
```

而当前 walking 问题的状态分布是：

```text
有前进速度
腿周期性摆动
接触状态不断切换
存在 yaw/lateral drift
```

所以更直接有效的是训练：

```text
walking residual stabilization
```

stand balance 可以作为备选：

```text
如果 residual walking 一训练就崩，再短暂训练 residual balance pretrain。
```

但当前不建议把它作为主线。

---

## 5. 实现阶段

## Stage 0：整理 open-loop gait 逻辑

### 目标

把 `disk_robot/scripts/open_loop_gait.py` 里的 gait 生成逻辑整理成可复用函数，供训练环境调用。

当前关键函数：

- `_leg_phase_offsets()`
- `_smooth_square()`
- `make_targets()`

建议迁移或复用到一个模块，例如：

```text
disk_robot/disk_robot/gait.py
```

不要让训练环境直接依赖 `scripts` 目录。

### 推荐接口

```python
def leg_phase_offsets(mode: str) -> dict[str, float]:
    ...


def smooth_square(phase: float, duty: float) -> float:
    ...


def make_open_loop_targets(
    neutral: np.ndarray,
    t: float,
    frequency: float,
    hip_amplitude: float,
    knee_amplitude: float,
    abd_amplitude: float,
    duty: float,
    mode: str,
) -> np.ndarray:
    ...
```

MJX/JAX 环境里可能还需要一个 JAX 版本：

```python
def make_open_loop_targets_jax(...):
    ...
```

或者先在环境初始化时固定 gait 参数，用 JAX 数组实现等价逻辑。

---

## Stage 1：在 MuJoCo smoke 环境里验证 residual 接口

### 目标

先不要直接上大规模 MJX 训练。先在本地 MuJoCo 环境里验证：

```text
ctrl = gait_target + residual
```

是否能稳定运行。

可以先用：

```text
residual = 0
```

确认新接口和原始 open-loop 行为一致。

然后测试简单手写反馈，例如：

```text
residual based on yaw_rate
residual based on lateral_velocity
```

目的是验证：

> 小幅 residual 是否真的能改善走歪问题。

### 简单反馈示例思想

如果机器人向左偏航，可以让左右腿产生轻微差动：

```text
left hip flex residual  += k * yaw_error
right hip flex residual -= k * yaw_error
```

或者根据侧向速度调整 abduction：

```text
left/right hip_abd residual 根据 lateral_velocity 做反向补偿
```

这不是最终控制器，但可以快速判断 residual 控制通道是否有效。

---

## Stage 2：设计 residual RL observation

为了让 policy 学会修正走歪，observation 里必须包含和漂移、偏航、步态相位相关的信息。

建议 observation 至少包含：

```text
1. torso orientation / projected gravity
2. linear velocity in body frame
3. angular velocity in body frame
4. joint positions
5. joint velocities
6. previous residual action
7. foot contacts
8. command forward velocity
9. gait phase sin/cos
10. yaw rate
11. lateral velocity
```

如果目标是沿世界 x 方向走，建议加入：

```text
heading error
```

也就是当前 torso heading 和目标 heading 的差。

### 为什么 gait phase 很重要

open-loop gait 是周期控制。policy 如果不知道当前处于 gait cycle 的哪个阶段，就很难判断：

- 哪条腿应该支撑；
- 哪条腿应该摆动；
- 当前 residual 应该加在哪条腿上；
- 某个脚接触异常到底是不是异常。

建议加入：

```text
sin(2π phase)
cos(2π phase)
```

如果后续支持不同 gait mode，也可以加入 desired contact pattern。

---

## Stage 3：设计 residual RL action

policy 输出仍然是 12 维，对应 12 个 actuator：

```text
fl_hip_abd
fl_hip_flex
fl_knee
fr_hip_abd
fr_hip_flex
fr_knee
hl_hip_abd
hl_hip_flex
hl_knee
hr_hip_abd
hr_hip_flex
hr_knee
```

action 语义改为：

```text
residual joint target
```

而不是完整关节目标。

控制公式：

```python
ctrl = gait_target + residual_scale * action
```

然后 clip 到 actuator control range。

初期建议：

```text
residual_scale = 0.03 或 0.05
```

如果 residual 学不动，再逐步增加到：

```text
0.08
0.10
0.15
```

不要一开始给太大。

---

## Stage 4：设计 reward

当前目标是“走直、走稳、不要破坏 open-loop gait”。

reward 应围绕以下方向设计。

### 正奖励

```text
+ forward velocity tracking
+ upright
+ target torso height
+ alive
```

### 关键惩罚

```text
- lateral velocity
- yaw rate
- heading error
- roll/pitch angular velocity
- disk contact
- termination
- residual action magnitude
- residual action delta
```

尤其是这三个和“走远了会歪”强相关：

```text
penalty_lateral_velocity
penalty_yaw_rate
penalty_heading_error
```

如果没有 heading error，policy 可能只会减少角速度，但已经歪掉以后不一定知道要转回来。

---

## Stage 5：训练策略

### 第一阶段：固定 gait 参数

先固定 open-loop gait 参数，不要同时优化 gait 参数和 RL。

例如固定：

```text
mode = trot
frequency = 1.2
hip_amplitude = 0.20
knee_amplitude = 0.12
abd_amplitude = 0.0
duty = 0.6
```

这样 RL 只需要学习 residual correction。

### 第二阶段：短 episode

一开始不要直接要求走很久。

可以从：

```text
2s
4s
6s
8s
```

逐渐增加 episode 长度。

因为当前问题是误差累积，episode 太长会让早期训练很难。

### 第三阶段：逐渐增加扰动

先无扰动训练，让 policy 学会基本闭环修正。再逐渐加入：

```text
初始 yaw perturbation
初始 lateral velocity
初始 angular velocity
摩擦轻微随机化
质量轻微随机化
gait 参数轻微随机化
```

这样可以增强鲁棒性。

### 第四阶段：增大速度或复杂度

在 residual 稳定后，再逐渐增加：

```text
command_velocity
frequency
gait amplitude
episode duration
terrain/contact variation
```

---

## 6. 和 stand balance 的关系

当前不把 stand balance 作为主线。

但可以保留一个可选 fallback：

### 如果 residual walking 训练失败

失败表现包括：

```text
policy 一开始就破坏 open-loop gait
episode 极短
机器人频繁摔倒
reward 完全不升
residual action 发散
```

可以短暂训练 residual balance：

```text
ctrl = stand_target + residual_policy(obs)
```

训练目标是：

```text
抗扰动恢复 upright 和 height
```

然后用这个 policy 初始化 walking residual：

```text
ctrl = gait_target + residual_policy(obs, phase)
```

但如果 direct residual walking 能训起来，就不需要这个阶段。

---

## 7. 推荐文件改动

### 7.1 新增或整理 gait 工具模块

建议新增：

```text
disk_robot/disk_robot/gait.py
```

内容包括：

```text
leg_phase_offsets
smooth_square
make_open_loop_targets
phase computation
default gait params
```

这样 `disk_robot/scripts/open_loop_gait.py` 和训练环境都可以复用同一套 gait 逻辑。

---

### 7.2 修改 MuJoCo walking 环境

可能涉及：

```text
disk_robot/disk_robot/walk_env.py
```

增加 residual mode：

```text
use_open_loop_gait: bool
residual_action_scale: float
gait_frequency
gait_hip_amplitude
gait_knee_amplitude
gait_abd_amplitude
gait_duty
gait_mode
```

step 中：

```python
base_ctrl = make_open_loop_targets(...)
ctrl = base_ctrl + residual_action_scale * action
```

---

### 7.3 修改 MJX/Brax 环境

可能涉及：

```text
disk_robot/disk_robot_mjx/brax_env.py
```

实现 JAX 版本的 gait target 生成。

核心变化：

```text
原来：
ctrl = neutral_ctrl + action_scale * action

改为：
gait_ctrl = make_gait_ctrl(time or step_count, gait_params)
ctrl = gait_ctrl + residual_action_scale * action
```

同时 observation 增加：

```text
phase_sin
phase_cos
possibly heading_error
```

---

### 7.4 修改配置文件

可能涉及：

```text
disk_robot/disk_robot/walk_config.py
```

增加字段：

```python
use_open_loop_gait: bool = True
residual_action_scale: float = 0.05

gait_frequency: float = 1.2
gait_hip_amplitude: float = 0.20
gait_knee_amplitude: float = 0.12
gait_abd_amplitude: float = 0.0
gait_duty: float = 0.6
gait_mode: str = "trot"

penalty_lateral_velocity: float = ...
penalty_yaw_rate: float = ...
penalty_heading_error: float = ...
```

需要注意：

- 如果 observation 增加 phase 或 heading error，`observation_size` 也要同步更新。
- smoke 环境和 MJX 环境要保持 observation 定义一致。

---

## 8. 验证路线

### Step 1：open-loop baseline

先记录当前 open-loop 表现：

```text
duration
dx
dy
final_z
yaw drift
lateral drift
是否摔倒
```

例如运行：

```bash
python disk_robot/scripts/open_loop_gait.py --duration 8 --mode trot
```

如果有 viewer：

```bash
python disk_robot/scripts/open_loop_gait.py --duration 8 --mode trot --viewer
```

---

### Step 2：residual = 0 等价测试

实现 residual 接口后，设置：

```text
residual_action = 0
```

验证结果应接近原始 open-loop。

如果 residual=0 时行为已经明显不同，说明 gait target 或 control pipeline 有 bug。

---

### Step 3：手写 residual feedback 测试

加入简单 yaw/lateral feedback，不训练。观察是否能减少：

```text
dy
yaw drift
摔倒率
```

如果简单反馈有效，说明 residual action 通道有价值。

---

### Step 4：短时 RL 训练

先训练短 episode：

```text
2s ~ 4s
```

目标不是最快速度，而是：

```text
比 open-loop 更直
比 open-loop 更不容易倒
residual action 不要太大
```

---

### Step 5：长时 rollout 对比

对比：

```text
open-loop
open-loop + trained residual
```

指标：

```text
dx
dy
dy/dx
yaw drift
final_z
termination rate
average forward velocity
average lateral velocity
average residual magnitude
```

关键不是只看 `dx`，而是看：

```text
单位前进距离的侧漂量 dy/dx
```

---

## 9. 重要风险

### 9.1 residual 太大会破坏 gait

解决：

```text
从 residual_scale = 0.03 或 0.05 开始
加入 action penalty
加入 action delta penalty
```

---

### 9.2 reward 只奖励前进，会学歪

解决：

```text
加入 lateral velocity penalty
加入 yaw rate penalty
加入 heading error penalty
```

---

### 9.3 policy 不知道 gait phase

解决：

```text
observation 加 sin/cos phase
```

---

### 9.4 stand balance pretrain 可能抑制行走

如果 stand pretrain reward 过度惩罚动作，policy 会偏保守。所以当前不优先走 stand pretrain。

---

### 9.5 open-loop gait 参数本身有系统性偏差

如果 gait 左右不对称、接触时机不好，RL residual 可能会一直补偿，训练负担大。

解决：

```text
先人工或简单搜索调 open-loop gait 参数
再训练 residual
```

---

## 10. 推荐实施顺序

最终建议路线：

```text
1. 保留当前 open_loop_gait.py 作为 baseline
2. 抽出 gait target 生成逻辑到可复用模块
3. 在 MuJoCo 环境中实现 residual 控制接口
4. residual=0 验证和 open-loop 行为一致
5. 加入 phase observation
6. 加入 lateral/yaw/heading 相关 reward
7. 在 MJX/Brax 环境中实现同样的 residual 控制
8. 用小 residual_scale 训练短 episode
9. 对比 open-loop 和 residual policy 的长距离直行表现
10. 成功后再逐渐增加 episode duration、速度和扰动
```

---

## 11. 一句话总结

当前不需要优先训练抗扰动 stand。

因为已经有：

```text
stand 可行
open-loop 可走
```

真正缺的是：

```text
walking 过程中的闭环稳定修正
```

所以最合理的主线是：

```text
open-loop gait 作为 base controller
RL policy 输出小幅 residual correction
用 reward 明确惩罚侧漂、偏航和姿态误差
```

这条路线比从零训练 walking 更稳，也比单独训练 stand balance 更贴近当前遇到的问题。
