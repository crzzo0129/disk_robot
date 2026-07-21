# 结构参数方案研究

> 本文是 [`structure_variant_study.md`](structure_variant_study.md) 的中文版。所有命令默认在 `disk_robot/` 目录下执行。

## 结论

当前暂定的行走/滚动结构候选方案为：

```text
hip_y       = +/-0.090 m
leg_scale   = 0.85
disk_radius = 0.200 m
controller  = Kp 10.0, Kd 0.4, torque limit 3 Nm
```

对应模型保存在 `assets/pupper_v3_disk_structure_candidate.xml`，目前是前进方向
Teacher-Student 训练的默认模型。原始可视化 XML 保留为未缩放的几何基准。

## 模型修正

圆盘圆柱体经过旋转，其对称轴位于机体 Y 轴。此前显式惯量把较大的对称轴转动惯量
错误地分配给了机体 X 轴。当前使用中的 XML 和候选 XML 已改为：

```xml
diaginertia="0.0158632 0.03012 0.0158632"
```

对于质量为 `m`、半径为 `r`、完整厚度为 `L` 的圆柱体：

```text
Iyy = 0.5 * m * r^2
Ixx = Izz = m * (3*r^2 + L^2) / 12
```

## 几何参数

这里的“扫描基准”是已经相对 Pupper 原始设计拉长过腿部的
`pupper_v3_disk_visual.xml`，并不是 Pupper 原始 XML。`leg_scale` 表示在该拉长基准上
继续乘以一个局部比例。

| 参数 | 扫描基准（disk visual） | 候选结构 |
|---|---:|---:|
| 髋关节半宽 | 0.070 m | 0.090 m |
| 站立时足端支撑宽度 | 0.188 m | 0.221 m |
| 站立时足端支撑长度 | 0.200 m | 0.200 m |
| 腿部运动学缩放比例 | 1.00 | 0.85 |
| 圆盘半径 | 0.200 m | 0.200 m |
| 质心相对基座的 X 坐标 | -0.025 m | -0.021 m |
| 质心相对基座的 Z 坐标 | -0.051 m | -0.048 m |

三份模型在零关节角下的右前腿髋部到足端距离为：

| 模型 | 零位 hip-to-foot 距离 | 相对 Pupper 原始设计 |
|---|---:|---:|
| Pupper 原始设计 XML | 0.11317 m | 1.000 |
| Pupper disk visual（扫描基准） | 0.16940 m | 1.497 |
| disk structure candidate（`leg_scale=0.85`） | 0.14399 m | 1.272 |

因此，`leg_scale=0.85` 的准确含义是：把已经拉长约 49.7% 的 disk visual 腿部回调到
其 85%。候选腿相对 Pupper 原始设计仍然拉长约 27.2%，不能笼统称为“比 Pupper
缩短了腿”。

当前的临时缩放方法保持由电机主导的连杆质量和足端球半径不变，仅缩放连杆偏移、
远端惯性位置，以及中段/末端连杆的可视化网格。该方法只用于早期结构筛选；最终的
质量、质心和惯量必须由修订后的 CAD 模型给出。

## 参数扫描

CPU 扫描共测试了 27 种组合：

```text
hip_y       = 0.070, 0.085, 0.090 m
leg_scale   = 1.00, 0.90, 0.85
disk_radius = 0.200, 0.180, 0.170 m
```

所有方案使用相同的、带启动渐变的 256 点小跑参考轨迹：

```text
frequency=0.8 Hz, stride=0.04 m, height=0.025 m, duty=0.72
Kp=10.0, Kd=0.4, torque limit=3 Nm, duration=8 s
```

| 方案 | 净前进速度 | 横滚/俯仰 RMS | 跟踪 RMSE | 力矩饱和率 |
|---|---:|---:|---:|---:|
| 当前尺寸 | 0.0201 m/s | 1.32 deg | 0.0546 rad | 0.0% |
| 腿长 0.90、髋宽 0.070 | 0.0253 m/s | 1.07 deg | 0.0497 rad | 0.0% |
| 腿长 0.85、髋宽 0.070 | 0.0273 m/s | 0.91 deg | 0.0476 rad | 0.0% |
| **腿长 0.85、髋宽 0.090** | **0.0277 m/s** | **1.07 deg** | **0.0479 rad** | **0.0%** |

在已经拉长的 disk visual 基准上，把腿长比例从 1.00 回调到 0.85 带来了可重复的性能
提升。候选腿相对 Pupper 原始设计仍然更长。单独增加髋宽并没有改善对角小跑，因为两足支撑时
支撑区域仍会退化成一条对角线。候选方案仍然保留更宽的髋距，以增加四足重叠支撑、
启动、低速行走和抗扰动时的稳定裕量。

行走时圆盘不接触地面，因此圆盘半径对上述行走结果几乎没有影响，不能只根据行走
得分选择圆盘半径。

## 折叠包络

折叠检查测量每个足端球在圆盘 X-Z 滚动平面内的径向最远点。负裕量表示足端突出到
圆盘轮廓之外。

| 结构 | 折叠径向裕量 |
|---|---:|
| 当前结构，半径 0.200 | -11.3 mm |
| 候选结构，半径 0.200 | -1.8 mm |
| 候选结构，半径 0.170 | -31.8 mm |

从 disk visual 拉长基准回调到 0.85 后，腿部几乎可以完全收入现有滚动圆周。把圆盘半径缩小到 `0.17 m` 会明显加重
滚动干涉，因此候选方案保留 `0.20 m` 半径。CAD 设计应提供至少数毫米的正折叠裕量；
剩余的 1.8 mm 干涉可通过调整折叠关节目标或末端零件设计消除。

## 测试与复现

### 1. 安装基础依赖

```powershell
cd C:\Users\12481\Desktop\OH-WorkSpace\robot_description\disk_robot
python -m pip install -r requirements.txt
```

如果系统命令是 `python3.12`，可把后续命令中的 `python` 替换为 `python3.12`。

### 2. 运行快速回归测试

只检查结构缩放、惯量、候选 XML 和默认基准是否正确：

```powershell
python -m pytest tests/test_structure_variants.py -q
```

同时检查结构、IK 参考和步态：

```powershell
python -m pytest tests/test_structure_variants.py tests/test_ik_reference.py tests/test_ik_gait.py -q
```

完整 CPU 测试：

```powershell
python -m pytest tests -q
```

### 3. 复现 27 组结构扫描

```powershell
python -m scripts.sweep_structure_variants --duration 8 --out docs/structure_sweep_results_retest
```

脚本会逐项打印 `vx`、横滚/俯仰、跟踪误差、力矩饱和和失败状态，最后按照综合分数
输出前 10 名，并生成：

```text
docs/structure_sweep_results_retest.csv
docs/structure_sweep_results_retest.json
```

预期第一名应接近：

```text
hip_y=0.090, leg_scale=0.85, disk_radius=0.200
score=0.02162, vx=0.0277 m/s, roll/pitch RMS=1.07 deg
```

由于 MuJoCo 版本和浮点计算差异，末位小数可能略有变化。若结果出现明显差异，先确认
扫描使用的是未缩放基准 `assets/pupper_v3_disk_visual.xml`，而不是已经缩放过的候选 XML。

### 4. 可视化候选结构

查看约 `0.08 m/s` 的标定步态：

```powershell
python scripts\view_ik_gait.py --xml assets\pupper_v3_disk_structure_candidate.xml --training-reference --neutral-pose model --mode trot --target-speed 0.08 --height 0.025 --duty 0.72 --ramp 0.5 --kp 10 --kd 0.4 --torque-limit 3 --phase 0 --duration 0
```

查看当前标定范围上限附近的 `0.10 m/s`：

```powershell
python scripts\view_ik_gait.py --xml assets\pupper_v3_disk_structure_candidate.xml --training-reference --neutral-pose model --mode trot --target-speed 0.10 --height 0.025 --duty 0.72 --ramp 0.5 --kp 10 --kd 0.4 --torque-limit 3 --phase 0 --duration 0
```

测试一个参数组合但不生成新 XML：

```powershell
python scripts\view_ik_gait.py --training-reference --hip-y 0.09 --leg-scale 0.85 --disk-radius 0.20 --neutral-pose model --mode trot --frequency 0.8 --stride 0.04 --height 0.025 --duty 0.72 --ramp 0.5 --kp 10 --kd 0.4 --torque-limit 3 --phase 0 --duration 0
```

观察时重点检查：机器人是否持续向正 X 方向移动、圆盘是否触地、机身是否明显摇摆、
足端是否打滑，以及关节动作是否出现突跳。

### 5. 重新生成候选 XML

只有在确认要更新候选结构时才执行：

```powershell
python -m scripts.write_structure_candidate
```

该命令默认从未缩放模型生成
`assets/pupper_v3_disk_structure_candidate.xml`，会覆盖现有候选 XML。只做预览或参数
对比时不需要执行它。

原始排序结果位于 `docs/structure_sweep_results.csv` 和
`docs/structure_sweep_results.json`。

## 下一阶段门槛

在候选结构满足以下条件前，不应开始新的长时间 PPO 训练：

1. 可视化结果显示持续为正的世界坐标系 `dx`，而不仅是周期性的机体坐标系速度。
2. 在硬件上限 `Kp=10` 时，关节跟踪误差和触地冲击仍处于可接受范围。
3. CAD 调整后，折叠足端不再超过圆盘滚动包络。
4. 使用更宽髋距后，后腿推蹬和滚动动作仍不存在碰撞问题。
5. 使用 CAD 计算得到的质量、质心和惯量替换临时缩放值。
