# Student 闭环模仿失败：从低 MSE 到 previous-action 自反馈的诊断记录

> 状态：T3--T8 的正式实验结论与可复用诊断方法。
>
> 更新日期：2026-07-18

## 1. 问题概述

固定 `vx=0.08 m/s` 的 privileged PPO Teacher 已通过名义和扰动验收，但多个 Student
都出现了同一种反常现象：离线 Behavior Cloning（BC）误差非常低，闭环运行却几乎不
前进，甚至向后移动。

最容易做出的反应是继续调 BC、DAgger、网络、学习率，或者推翻整个
Teacher--Student pipeline。最终实验表明，这些都没有击中主要矛盾。真正的问题是：

> T5 Student 利用了 observation 中高度相关的 previous-action history，学到了一个
> 在 Teacher-forced 数据上 MSE 很低、但部署时会递归放大自身误差的捷径。

删除 previous action 后，保持 Teacher、标签、轨迹、网络规模、训练步数和验收条件
不变，T8 Student 通过了名义和扰动验收。因此这是一个经过单变量消融确认的因果结论，
不只是相关性猜测。

## 2. 先区分三种“历史”

这里的 previous action 不是 observation history 的同义词。

- **previous-action history**：过去由策略输出的关节动作。它是策略自己的输出，下一步
  又成为策略输入，因此能形成直接的自反馈回路。
- **sensor history**：关节位置、关节速度、IMU、机身速度等过去测量。它描述机器人的
  真实动力学响应，可用于估计趋势、延迟和部分可观测状态。
- **controller phase**：`sin(phase), cos(phase), gait_blend`。它由控制器内部时钟维护，
  用于消除周期动作在相似物理观测下的标签歧义，不是足部接触或 privileged 信息。

T8 只删除了四帧中每帧的 12 维 previous action（共 48 维），保留了四帧 sensor history
和 3 维 phase：

```text
T5: 4 * (36 sensor/command + 12 previous action) + 3 phase = 195
T8: 4 * 36 sensor/command                        + 3 phase = 147
```

所以当前证据支持“删除 previous action”，并不支持“删除全部历史”。是否缩短传感历史
需要另外做 4 帧、2 帧、1 帧的受控消融。

## 3. previous action 为什么最初看起来合理

在机器人控制中加入上一动作通常有合理目的：

1. 补充执行器延迟和位置环内部状态；
2. 帮助策略理解当前关节目标从哪里来；
3. 生成更平滑的动作，减少相邻控制步跳变；
4. 在 on-policy RL 中改善近似 Markov 性。

问题不在于 previous action 永远不能使用，而在于它与当前的离线蒸馏方式产生了训练--部署
不一致。Teacher 数据集中的输入是 `a^T_(t-1)`，标签是 `a^T_t`；部署时输入变成
`a^S_(t-1)`。平滑周期步态中相邻 Teacher 动作高度相关，因此 Student 很容易通过复制或
外推 `a^T_(t-1)` 获得极低的一步 MSE，而不必真正依赖 phase、关节状态和 IMU 推断当前
应该输出什么。

BC 训练使用 Teacher forcing：每一个训练样本都带着正确的 Teacher 历史。训练目标只要求
一步预测准确，并不惩罚 Student 自己的小误差在随后几十步中造成的影响。因此“更低的
离线 MSE”可能恰好意味着模型更强地依赖了这个捷径。

## 4. 闭环中为什么会爆炸

部署后的误差可以局部写成：

```text
e_t ~= J_prev * e_(t-1) + J_state * delta_state_t + one_step_error_t
```

其中 `J_prev` 表示 Student 输出对 previous-action 输入的局部增益。Teacher 轨迹上的第一步
误差虽然只有约 `0.002`，它会作为下一步 previous action 的偏差重新进入网络。如果这个
回路的有效增益大于 1，误差会在几步内快速增长；动作偏差又改变关节状态，进一步把系统
推出训练分布。

这解释了两个看似矛盾的现象：

- 离线数据上 action RMSE 只有 `0.0020543`；
- 闭环第 2--3 步 Student--Teacher action error 已增长到 `0.11323/0.19855`。

Teacher 本身并不脆弱。向 Oracle 动作加入 RMS 为 `0.002` 甚至 `0.01` 的独立高斯噪声，
机器人仍稳定前进。危险的不是同样大小的一次性误差，而是与策略输出相关、每一步重新
注入并被网络高增益放大的结构化误差。

## 5. 诊断决策链

### 5.1 先确认失败是真的，不把训练 loss 当闭环指标

T3 和 T5 都获得了非常低的 BC loss，但闭环速度严重下降：

| 策略 | 离线 loss | 名义 `vx` | 结论 |
|---|---:|---:|---|
| accepted Teacher | 不适用 | `0.0817` 左右 | 基准通过 |
| T3 phase-free BC | `0.0000060` | `0.0290` | 闭环失败 |
| T5 phase-conditioned BC | `0.0000045` | `-0.0076` | 闭环失败 |

这一步得到的结论只是“存在训练分布与闭环分布差异”，还不能直接断言原因是普通 covariate
shift、phase、动作接口或 Teacher 不可模仿。

### 5.2 用 Oracle direct-action 排除接口和标签错误

`scripts.diagnose_phase_student` 在同一状态上比较：

```text
Teacher residual control    vx= 0.0817
Oracle direct action        vx= 0.0817
Learned Student             vx=-0.0071
```

把 Teacher 在线产生的完整位置标签直接送进 Student actuator 路径可以完整保留步态。这排除了：

- residual 到 full-position action 的转换错误；
- Student actuator 接口定义错误；
- phase 与标签错一控制步；
- Teacher 行为本身无法由 direct action 表达。

因此没有理由推翻 Teacher 或动作接口，问题收缩到“学习到的 Student 函数如何在闭环被调用”。

### 5.3 用动作容差实验排除脆弱轨迹假说

`scripts.audit_phase_student_failure` 对 Oracle 动作加入 bias 和 Gaussian noise。正式结果中：

```text
baseline                    vx=0.0816
bias RMS=0.002              vx=0.0795
bias RMS=0.010              vx=0.0657
gaussian RMS=0.010          vx=0.0831
all tested failure_rate=0
```

如果 `0.002` 的普通动作误差足以让 gait 崩溃，那么可以把问题归结为 Teacher 轨迹没有稳定
裕度。但实验否定了这一点。Student 的 `0.002` 离线误差必须通过某种闭环结构被放大。

### 5.4 做逐步 paired divergence，定位误差何时开始

从完全相同的 reset 同时运行 Oracle 和 Student，前十步误差为：

```text
step  student_teacher  label_drift  q_rmse
0     0.00206          0.00000      0.00045
1     0.01951          0.00175      0.00249
2     0.11323          0.01673      0.01195
3     0.19855          0.04631      0.03411
4     0.26411          0.05147      0.06139
```

误差在第 1--2 步就开始爆炸，早于明显的姿态崩坏；同时 Teacher label drift 远小于
Student--Teacher error。因此主要原因不是 Student 进入新状态后 Teacher 标签发生巨大冲突，
而是 Student 策略自身对很小的输入偏移极其敏感。

### 5.5 分离 Oracle trajectory 与 Student trajectory

`scripts.diagnose_phase_student_feedback` 把同一个 Student 放在两类输入上：

- Oracle trajectory：始终输入 Teacher/Oracle 产生的干净历史；
- closed-loop trajectory：输入 Student 自己造成的历史。

T5 在 Oracle trajectory 上持续保持约 `0.002--0.006` 的误差，证明网络容量和监督拟合没有
问题；在自己的 trajectory 上，第 2--3 步误差迅速上升到 `0.09--0.21`。这直接复现了
teacher forcing 与部署方式之间的差异。

### 5.6 用 observation-group 反事实定位反馈通道

诊断脚本逐组把 closed-loop observation 替换为同一时刻的 Oracle observation，并测量策略
输出恢复多少：

```text
all previous-action history recovered fraction  0.901
latest previous action recovered fraction       0.547
all previous-action Jacobian spectral gain      59.832
latest previous-action Jacobian spectral gain   22.668
joint-velocity-history Jacobian spectral gain    0.203
```

仅替换 previous-action history 就能消除 90.1% 的策略漂移，而且局部增益远大于 1。这把
“某种 covariate shift”进一步定位为明确的 previous-action 自反馈通道。

### 5.7 最后用单变量结构消融确认因果

T8 仅从 Student observation 中结构性删除 previous action。它不是在训练时随机 mask，也不是
部署时填零；网络从一开始就没有这个输入。其余关键变量保持与 T5 一致。

正式结果：

| 指标 | T5 | T8 |
|---|---:|---:|
| Student observation | 195 | 147 |
| final BC loss | `0.0000045` | `0.0000164` |
| nominal `vx` | `-0.0076` | `0.0823` |
| disturbed `vx` | `-0.0092` | `0.0785` |
| nominal failure | `0.004` | `0.000` |
| disturbed failure | `0.004` | `0.000` |
| disturbed recovery | `1.595 s` | `0.413 s` |
| accepted | False | True |

T8 的离线 loss 比 T5 更高，但闭环表现几乎完整保留 Teacher。这正是预期：删掉捷径后，
Student 必须使用 phase、关节状态、关节速度和机体反馈完成预测，一步拟合稍难，却获得了
稳定的闭环函数。

T8 的反馈复查也没有再出现递归爆炸：前十步 closed-loop error 保持在约
`0.0025--0.0102`，policy shift 约 `0.0017--0.0051`，`q_rmse` 到第 9 步仍只有 `0.00347`。

## 6. 为什么没有推翻现有 pipeline

这次一度有理由怀疑 Teacher--Student 路线本身，因为 BC 和 DAgger 都失败了。但是否换
pipeline 应由故障边界决定：

1. accepted Teacher 在名义和扰动环境都通过，说明行为源有效；
2. Oracle direct action 完整复现 Teacher，说明动作表示和执行接口有效；
3. Student 在 Oracle trajectory 上拟合准确，说明网络容量与标签基本有效；
4. 动作容差实验说明 Teacher gait 有足够局部稳定裕度；
5. group counterfactual 将故障定位到一个可移除的 observation 通道；
6. 单变量消融后原 pipeline 立即通过。

因此失败发生在 Student observation contract，而不是 Teacher、IK、动作接口、BC 算法整体或
Teacher--Student 思路本身。推翻整条 pipeline 会同时丢掉已经被实验验证的部分，也无法解释
为什么 T8 只改一个输入就成功。

DAgger 没有救回 T5 也不矛盾。它处理的是 Student 访问状态与 Teacher 数据状态之间的分布
偏移，但当策略内部存在高增益自反馈捷径时，少量 on-policy 数据和监督更新未必能改变这个
更容易降低 MSE 的表示。先修正 observation contract 比继续扩大 DAgger 更直接。

## 7. 可复用的调试原则

以后遇到“离线很好、闭环失败”，按以下顺序排查：

1. **确认指标语义**：分别看离线 supervised error、closed-loop task metric 和失败时序。
2. **建立 Oracle**：绕过 learned policy，直接执行在线标签，验证动作与执行接口。
3. **测局部裕度**：给 Oracle 加独立 bias/noise，判断任务是否本来就极端脆弱。
4. **paired rollout**：相同 reset 下逐步比较 Oracle 和 learned policy，不只比较 episode 均值。
5. **拆分 policy error 与 label drift**：不要把两者都笼统称为 covariate shift。
6. **按输入组做反事实替换**：找出哪个 observation group 真正驱动策略漂移。
7. **测局部增益**：高相关输入不一定危险；高相关、可自反馈且增益大于 1 才是强证据。
8. **做单变量消融**：保持数据源、seed、训练量和验收不变，验证因果。
9. **以闭环结果选模型**：更低的一步 MSE 不是更好控制器的充分条件。

## 8. 当前边界与下一风险

T8 只证明固定 `vx=0.08 m/s`、无 previous-action 输入的 phase-conditioned Student 可以保留
Teacher。它尚未证明：

- 可以删除四帧 sensor history；
- 可以在没有机身线速度估计的实机上工作；
- 可以响应连续变化的手柄 command；
- 可以泛化到停止、后退、横移和偏航。

尤其是当前数据中的 command 固定不变。T8 反馈诊断中 command observation 没有产生轨迹
差异，因此即使网络对 command 的局部 Jacobian 较大，也不能说明它学会了正确的速度响应。
接入手柄前必须训练 command-conditioned Teacher/Student，并做速度 sweep、停止/启动过渡和
command 反事实诊断。不要把未训练的 joystick 数值直接送入当前固定速度策略。
