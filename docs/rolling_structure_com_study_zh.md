# 圆盘机器人滚动结构与质心实验

## 实验目的

本实验把滚动性能从行走评分中独立出来，考察以下参数对轴向滚动的影响：

- 髋关节半宽 `hip_y`；
- 相对 disk visual 拉长基准的腿部比例 `leg_scale`；
- 圆盘半径 `disk_radius`；
- 圆盘本体惯性质心相对圆盘几何中心的 X/Z 偏置。

这里的“圆盘质心”只指 `base_link` 中质量为 `1.506 kg` 的圆盘本体惯性质心，并不等于
包含四条腿后的整机质心。实验同时计算 `rolling_folded` 姿势下的整机质心位置。

## 实验方法

扫描脚本为 `scripts/sweep_rolling_variants.py`，使用未缩放的
`assets/pupper_v3_disk_visual.xml` 作为结构基准，并动态应用结构参数。

粗扫描包含 72 组：

```text
hip_y       = 0.070, 0.090 m
leg_scale   = 1.00, 0.85
disk_radius = 0.170, 0.200 m
disk_com_x  = -0.030, 0.000, +0.030 m
disk_com_z  = -0.030, 0.000, +0.030 m
```

每组方案进行三次测试：

1. 以 `+0.80 m/s` 初始轮缘/线速度向前滚动 4 秒；
2. 以 `-0.80 m/s` 初始轮缘/线速度反向滚动 4 秒；
3. 从静止释放 2 秒，测量偏心引起的自行滚动。

角速度按 `omega = v / radius` 计算，因此不同圆盘半径具有相同初始线速度。正反双向
结果取平均，并惩罚方向不对称，避免 X 向偏心天然偏向某一方向而得到虚高分数。

主要指标包括：

- 4 秒双向平均滑行距离和最终速度；
- 纯滚动约束滑差 `|vx - radius * wy|`；
- 侧向漂移和圆盘轴倾角；
- 圆盘接地率和足端触地率；
- 静止释放漂移；
- `rolling_folded` 姿势的足端径向包络；
- 整机质心相对圆盘轴心的径向偏差。

## 主要结果

### 1. 当前行走候选的圆盘质心需要上移

当前行走候选结构为：

```text
hip_y=0.090 m, leg_scale=0.85, disk_radius=0.200 m
```

| 圆盘本体 COM 偏置 `(x,z)` | 整机 COM 径向偏差 | 4秒平均距离 | 双向距离差 | 滑差 RMS | 静止漂移 |
|---|---:|---:|---:|---:|---:|
| `(0, 0) mm` | 13.37 mm | 2.626 m | 2.4 mm | 0.0373 m/s | +51 mm |
| `(-5, +30) mm` | **1.15 mm** | **3.414 m** | **4.4 mm** | 0.0465 m/s | -36 mm |
| `(-5, +35) mm` | 3.43 mm | 3.516 m | 3.5 mm | 0.0471 m/s | -40 mm |

把圆盘本体质心上移约 30 mm、向后移动约 5 mm，可抵消收腿后腿部质量造成的整机
质心偏置。`(-5,+30) mm` 使整机质心距离圆盘轴心仅约 `1.15 mm`，同时把滑行距离
从 `2.626 m` 提高到 `3.414 m`，增幅约 30%。

`(-5,+35) mm` 的综合扫描分数和距离略高，但整机质心已经越过轴心。考虑 CAD 误差、
静止稳定性和双向一致性，当前更推荐 `disk_com_x=-0.005 m、disk_com_z=+0.030 m` 作为
下一版结构候选，而不是继续追求更高的 Z 偏置。

### 2. 腿长 1.00 的纯滑行距离更远，但不一定是整机最优

粗扫描纯滚动第一名为：

```text
hip_y=0.090 m, leg_scale=1.00, disk_radius=0.200 m
disk_com=(0,+0.030) m
```

其4秒双向平均滑行距离为 `3.634 m`，比推荐的 `leg_scale=0.85` 折中方案高约 6.4%。
但它的 `rolling_folded` 足端径向裕量只有 `27.7 mm`，而 `leg_scale=0.85` 为
`60.0 mm`；并且此前行走实验显示 `leg_scale=0.85` 的行走速度和跟踪性能更好。

因此，当前证据不支持仅为增加约 6% 的被动滑行距离就恢复到 `leg_scale=1.00`。如果
目标是一台同时行走和滚动的机器人，保留 `0.85` 更稳妥。

### 3. 髋宽对直线轴向滑行几乎没有影响

`hip_y=0.070 m` 与 `0.090 m` 在轴向滑行中的距离和稳定性几乎相同。这符合预期：腿部
已经收起，且圆盘沿 Y 轴滚动，髋宽主要影响行走支撑和侧向扰动，而不是理想平地直线
滑行。髋宽应继续由行走、启动和抗扰动实验决定。

### 4. 半径 0.17 m 可以滚动，但综合上不如 0.20 m

在相同初始线速度下，`leg_scale=0.85、COM=(0,+30 mm)` 时：

| 圆盘半径 | 4秒平均距离 | 滑差 RMS | 足端径向裕量 |
|---|---:|---:|---:|
| 0.170 m | 3.396 m | 0.055 m/s | 30.0 mm |
| 0.200 m | 3.413 m | 0.046 m/s | 60.0 mm |

小圆盘距离只略低，但滑差更高、收腿包络裕量减半。当 `leg_scale=1.00` 时，半径
`0.17 m` 的足端还会超出滚动圆周约 `2.3 mm`。当前保留 `0.20 m` 更合理。

## 当前推荐

兼顾已有行走结果、收腿包络和本次被动滚动实验，推荐进入下一轮启动滚动实验的参数为：

```text
hip_y       = +/-0.090 m
leg_scale   = 0.85          # 相对 disk visual；相对原始 Pupper 仍为拉长
disk_radius = 0.200 m
disk_com_x  = -0.005 m      # 相对圆盘几何中心
disk_com_z  = +0.030 m
rolling pose = rolling_folded
```

该方案在当前仿真中的关键结果：

```text
complete COM radial offset = 1.15 mm
mean coast distance        = 3.414 m / 4 s
forward/reverse difference = 4.4 mm
final speed                = 0.860 m/s
slip RMS                   = 0.0465 m/s
lateral drift              = 0.65 mm
axis tilt RMS              = 0.015 deg
disk contact fraction      = 98.0%
foot contact fraction      = 0.0%
rest drift                 = -36 mm / 2 s
failure                    = false
```

## 复现命令

### 交互查看推荐方案

前向滚动：

```powershell
python -m scripts.view_rolling_variant
```

反向滚动和静止释放：

```powershell
python -m scripts.view_rolling_variant --direction reverse
python -m scripts.view_rolling_variant --direction rest
```

窗口默认运行8秒；使用 `--duration 0` 可持续运行到手动关闭窗口。

从 `disk_robot/` 目录执行单元测试：

```powershell
python -m pytest tests/test_rolling_structure_sweep.py -q
```

复现72组粗扫描：

```powershell
python -m scripts.sweep_rolling_variants --duration 4 --rest-duration 2 --initial-speed 0.8 --out docs/rolling_structure_com_sweep
```

复现当前候选结构附近的16组质心细扫：

```powershell
python -m scripts.sweep_rolling_variants --hip-y 0.09 --leg-scale 0.85 --disk-radius 0.20 --com-x -0.01 -0.005 0 0.005 --com-z 0.02 0.025 0.03 0.035 --duration 4 --rest-duration 2 --initial-speed 0.8 --out docs/rolling_candidate_com_refinement
```

原始结果：

- `docs/rolling_structure_com_sweep.csv/json`：72组粗扫描；
- `docs/rolling_candidate_com_refinement.csv/json`：16组质心细扫。

## 局限与下一步

本实验评价的是给定起步速度后的滚动保持能力，还没有证明机器人能靠后腿推蹬自行达到
该速度。仿真使用理想平地、固定 `rolling_folded` 姿势、刚性位置控制和临时质量参数；
绝对执行器做功不能直接作为硬件能耗结论。

下一阶段应使用同一组候选参数进行：

1. 从静止折叠并以后腿推蹬启动；
2. 固定推蹬能量而不是固定初始速度，比较启动后的距离；
3. 加入摩擦系数、地面坡度和 CAD 质量误差扰动；
4. 检查圆盘本体质心偏置在旋转全过程中的周期性重力扰动；
5. 在硬件力矩上限下测量启动成功率、速度、滑差和停止距离。
