# Forward Teacher-Student Pipeline

> 状态：当前可执行训练规范。固定 `vx=0.08 m/s` 的 T2a Teacher 和 T8 Student 均已通过。
>
> 更新日期：2026-07-18

## 1. 当前结论

当前已验证的主线是：

1. 从 XML `stand` keyframe 构造对称 trot IK reference；
2. 用 privileged residual PPO 训练抗扰 Teacher；
3. 用 Teacher 的在线完整位置动作做 Behavior Cloning；
4. Student 使用 controller-owned phase，但不使用 IK、接触信息、privileged observation 或
   previous action；
5. 只部署 Student。

冻结的正式产物：

```text
Teacher run: mjx_runs/teacher_t2a_seed0
Teacher step: 1,024,000
Student run: mjx_runs/student_t8_phase_bc_no_previous_action_seed0
Student policy: student_policy_phase_bc_no_previous_action.npz
```

T8 已证明当前 Teacher--BC pipeline 可以工作，不需要继续用 DAgger 修补固定速度 Student。
T3--T7 的失败证据链和 previous-action 根因见
[student_imitation_failure_debugging.md](student_imitation_failure_debugging.md)。

当前策略只在固定 `vx=0.08 m/s` 数据上训练。它还不是手柄控制策略，不能把未训练的任意
`[vx, vy, wz]` 直接送入当前 artifact。

## 2. 模型、IK 和动作契约

默认训练模型：

```text
assets/pupper_v3_disk_structure_candidate.xml
```

当前结构和控制参数：

```text
hip_y       = +/-0.090 m
leg_scale   = 0.85
disk_radius = 0.200 m
Kp          = 10.0
Kd          = 0.4
torque      = 3 Nm
```

固定速度参考：

```text
command_vx = 0.08 m/s
frequency  = 1.2 Hz
stride     = 0.0742135 m
height     = 0.025 m
duty       = 0.72
```

Teacher 输出 IK 周围的小残差：

```text
q_target = q_ik(phase) + residual_scale * tanh(teacher_policy)
```

Student 不读取 IK，输出相对 `stand` 的完整位置动作：

```text
q_target = q_stand + student_action_scale * tanh(student_policy)
```

Student runtime 不需要 Teacher residual、期望接触或足部接触传感器。

## 3. 当前 observation 契约

### 3.1 Teacher

冻结的 T2a Teacher observation 为 231 维。它包含原始 192 维四帧历史，以及 phase、接触、
IK tracking、previous residual、push、motor-strength 和 delay 等 privileged 信息。不得修改
该 observation 后继续加载旧 Teacher 参数。

### 3.2 T8 Student

每帧 deployable observation 为 36 维，堆叠四帧后为 144 维：

```text
body angular velocity          3
projected gravity              3
body linear velocity           3
joint position - q_stand      12
joint velocity                12
command [vx, vy, wz]           3
```

然后追加 3 维 controller-owned phase：

```text
sin(phase), cos(phase), gait_blend
```

总计：

```text
4 * 36 + 3 = 147
```

previous action 已从 Student 网络输入中结构性删除，不是置零或随机 mask。原因是旧 T5
Student 在 Teacher-forced BC 中把它当作标签捷径，部署时形成高增益递归误差。

不要把 previous action 与全部 observation history 混为一谈。当前四帧关节、IMU 和速度
历史仍保留，用于描述真实动力学响应；删减传感历史需要单独做 4/2/1 帧消融。

phase 来自控制器内部时钟，不来自足端接触。当前固定速度下以 `1.2 Hz` 推进；停止时 phase
冻结，`gait_blend` 向 stand 过渡。实机对应实现位于 `disk_robot/phase_clock.py`。

当前 `body linear velocity` 来自仿真真值。实机部署前必须提供等价状态估计，或者通过
加噪、蒸馏或新的 observation 消融降低这项依赖。

## 4. 已验证阶段

### T1b：名义行为保持

零初始化 residual actor、低通 residual 和 preserve selection 使 PPO 不再破坏 IK。T1b
通过名义保持，但只证明 PPO 能学习接近 no-op 的 residual。

### T2a：抗扰 privileged Teacher

T2a 加入 root-velocity push、motor-strength variation、command delay 和 reset perturbation，
并用配对 seed 同时验证名义保持与扰动改善。正式 run：

```text
selected_source=ppo
selected_step=1,024,000
nominal_preserved=True
disturbed_improved=True
accepted=True
```

这是当前唯一冻结的 Teacher 数据源。不要重训或覆盖它。

### T3--T7：定位 Student 闭环失败

- T3 phase-free BC：离线 loss 极低，闭环速度下降；
- T4 DAgger：标签 phase 歧义与更新导致策略趋向站立；
- T5 加 phase：消除周期标签歧义，但闭环仍失败；
- Oracle direct action：证明标签转换和 actuator 接口正确；
- T6 phase DAgger：未恢复运动；
- T7 paired audit：定位到 previous-action 自反馈捷径。

不要再把“更多 BC/DAgger 更新”作为这个已解决故障的默认处理。先检查 observation 是否包含
Teacher-forced、部署时由 Student 自己生成的递归输入。

### T8：删除 previous action，通过

正式结果：

```text
final BC loss                    0.0000164
nominal mean_velocity_x          0.0823
nominal failure_rate             0.000
nominal roll/pitch RMS           0.2120
disturbed mean_velocity_x        0.0785
disturbed failure_rate           0.000
disturbed post-push error        0.0338
disturbed recovery               0.413 s
accepted                         True
```

T8 的 loss 高于失败的 T5，但闭环保留了 Teacher。这是当前模型选择必须以闭环 retention
为准、不能只按一步 MSE 排序的直接证据。

## 5. 当前可复现实验

4090 节点没有可用 EGL 时使用 `--mujoco-gl disable`。

T8 技术 smoke：

```bash
python -m scripts.distill_phase_student_no_previous_action --teacher-run mjx_runs/teacher_t2a_seed0 --smoke --save-dataset --mujoco-gl disable --out mjx_runs/student_t8_phase_bc_no_previous_action_smoke_seed0
```

T8 正式复现：

```bash
python -m scripts.distill_phase_student_no_previous_action --teacher-run mjx_runs/teacher_t2a_seed0 --save-dataset --strict-acceptance --mujoco-gl disable --out mjx_runs/student_t8_phase_bc_no_previous_action_seed0
```

闭环反馈诊断：

```bash
python -m scripts.diagnose_phase_student_feedback --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t8_phase_bc_no_previous_action_seed0
```

历史失败策略的完整 audit：

```bash
python -m scripts.audit_phase_student_failure --teacher-run mjx_runs/teacher_t2a_seed0 --student-run mjx_runs/student_t5_phase_bc_seed0 --mujoco-gl disable
```

## 6. 验收原则

必须分别评估 nominal 和 disturbed rollout，并至少检查：

- mean forward velocity 与 `0.08 m/s` command 的误差；
- failure rate、disk contact 和 roll/pitch rate；
- lateral velocity 与 yaw rate；
- disturbed post-push error、recovery time 和 displacement；
- Student 相对同 seed Teacher 的 retention delta；
- rollout 初始若干步是否存在 action error 和 joint-state error 的递归放大。

`--strict-acceptance` 未通过时会保留报告和策略，但以退出码 2 结束。失败 artifact 只能用于
诊断，不能因为 supervised loss 较低而作为部署候选。

## 7. 下一阶段：command-conditioned 手柄控制

T8 的 30 秒长时复查表明，其闭环 action error 始终约为 `0.002`，没有再次出现 T5
的递归放大；但相对 Teacher 的额外横漂在 `9.8 s` 后超过 `0.02 m`，到 30 秒达到
`0.1952 m`。因此冻结 T8，不再为单一 `vx=0.08` 做 BC/DAgger 调参。后续速度网格除原有
500-step 指标外，必须增加至少 1,500 control steps 的直线性、yaw 和 T8 retention 门。

T9 的 episode-fixed 第一阶段已经实现。速度锚点为 `0/0.04/0.06/0.08/0.10 m/s`，新
Teacher 使用 command-conditioned IK bank 重新训练；T9 Student 输入为四帧 33 维纯物理
历史、一个当前 command 和 phase/blend，共 138 维。Teacher aggregate 训练结果不能直接
验收，必须先通过逐速度 nominal/disturbed 与 1,500-step grid gate。第一阶段不包含
within-episode stop/start，也不使用 DAgger。

下一阶段不是直接连接手柄，而是训练 command-conditioned Teacher/Student：

1. 先扩展一维 `vx` 范围和停止/启动；
2. 训练和采集时真实随机化 command，而不是只保留一个形式上的 command 输入；
3. 保持 T8 的 no-previous-action 契约；
4. command 建议只输入当前值，不必在四帧中重复；
5. 做速度 sweep、command counterfactual、切换过渡和延迟/噪声评估；
6. 一维通过后再加入 yaw，最后加入 `vy`。

当前 T8 固定 command 数据不能证明 command Jacobian 代表正确控制规律。只有 command 网格
闭环通过后，才可以接入手柄映射。

## 8. 兼容性规则

XML、关节顺序、observation、action 定义、reward、控制增益或网络结构任一改变，都必须新建
run 并重新验证数据集和 checkpoint。特别是：

- T2a Teacher 必须保持 231 维输入；
- T8 Student policy 必须保持 147 维输入；
- T5 的 195 维 normalizer/policy 不能与 T8 混用；
- 不得通过 padding、截断或填零伪装 observation 兼容。
