# Pupper XML 与 Disk Robot XML 几何参数对比

本文对比两个 MuJoCo XML 中编码的机器人参数：

- Pupper 风格圆盘机身模型：`disk_robot/assets/pupper_v3_disk_visual.xml`
- 历史抽象训练模型：`disk_robot/assets/disk_quadruped_extreme_train.xml`
- 当前训练模型：`disk_robot/assets/pupper_v3_disk_structure_candidate.xml`

> 本文中的尺寸表用于解释 Pupper 与早期抽象 disk 模型的几何差异，不再定义训练入口。Teacher-Student 训练默认加载结构候选 XML。

我也顺手检查了 `disk_robot/assets/disk_quadruped_extreme.xml`。它与训练版最相关的几何差别是：非训练版髋关节锚点使用 `x = +/-0.08 m`，训练版使用 `x = +/-0.10 m`。

下文所有距离单位都是米。Pupper 的腿部 body 使用了多层嵌套 quaternion，所以有效的髋关节、膝关节和足端位置是沿 XML 层级做坐标变换算出来的，不能只直接读局部 `pos` 字符串。

## 总体结论

历史 `disk_quadruped_extreme_train.xml` 不是 Pupper 的简单放大版。它是一个更简化、圆盘更大、腿更长、髋关节相对圆盘中心更低、腿链更接近竖直直链的模型。

与 `pupper_v3_disk_visual.xml` 相比：

- disk robot 躯干圆盘半径是 `0.24 m`，Pupper disk visual 的圆盘半径是 `0.14 m`。
- disk robot 躯干总厚度是 `0.09 m`，Pupper disk visual 的总厚度是 `0.07 m`。
- disk robot 的髋关节锚点在圆盘中心下方 `0.10 m`；Pupper 的髋关节锚点大致在 base body 原点高度附近。
- disk robot 名义 hip-to-foot 竖直距离是 `0.30 m`；Pupper 经过坐标变换后的零位 hip-to-foot 向量长度约 `0.114 m`，它的两段建模 offset 总和约 `0.174 m`。
- disk robot 使用简单的 capsule、sphere、cylinder 几何，并通过 density 推断质量；Pupper 使用显式惯量，并使用 CAD mesh 做视觉显示。

## 躯干/机身对比

| 参数 | Pupper disk visual | Disk robot train | 说明 |
|---|---:|---:|---|
| 主 body 名称 | `base_link` | `disk_torso` | 命名体系不同。 |
| 主碰撞形状 | cylinder | cylinder | 这两个对比文件里都是圆盘状。 |
| 圆柱半径 | `0.14` | `0.24` | disk robot 半径约为 Pupper 的 `1.71x`。 |
| 圆柱半厚度 | `0.035` | `0.045` | MuJoCo cylinder 的 `size` 是 `radius halfheight`。 |
| 总厚度 | `0.07` | `0.09` | disk robot 约厚 `1.29x`。 |
| body XML `pos` | `0 0 0.13` | `0 0 0.32` | 这是 `worldbody` 中 body 的放置位置；keyframe 还会设置 root pose。 |
| stand/root keyframe z | `0.28` | `0.408` | `disk_quadruped_extreme_train.xml` 的 `stand` 和 `walk_stand` 都是 `z=0.408`。 |
| body 惯量 | 显式质量 `1.506` | body 上无显式 inertial | disk robot 质量主要由 geom/density 推断。 |

原始 Pupper MJX 文件 `pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml` 使用的是简化 box 碰撞机身，而不是圆盘：

```text
base collision half-size = 0.04507 0.06379 0.129715
base collision pos       = 0.02146 0 0.03345
base mass                = 1.506
```

因此，`pupper_v3_disk_visual.xml` 更准确的理解是：把 Pupper 腿部布局接到了一个圆盘状机身上，而不是原始 Pupper 躯干本身。

## 髋关节锚点相对躯干的位置

下表坐标均表示为相对 torso/base body 原点的位置。

| 腿 | Pupper 髋关节相对 base | Disk robot train 髋关节相对 disk | 主要差异 |
|---|---:|---:|---|
| 右前腿 | `(0.075, -0.0835, 0.000)` | `(0.100, -0.075, -0.100)` | disk hip 更靠前，而且低很多。 |
| 左前腿 | `(0.075, 0.0835, 0.000)` | `(0.100, 0.075, -0.100)` | 同上。 |
| 右后腿 | `(-0.075, -0.0725, 0.000)` | `(-0.100, -0.075, -0.100)` | disk hip 更靠后，而且低很多。 |
| 左后腿 | `(-0.075, 0.0725, 0.000)` | `(-0.100, 0.075, -0.100)` | 同上。 |

从俯视平面看，髋关节相对圆盘半径的位置如下：

| 模型 | 前腿髋关节距中心半径 | 后腿髋关节距中心半径 | 圆盘半径 | 髋关节半径/圆盘半径 |
|---|---:|---:|---:|---:|
| Pupper disk visual | `0.112` | `0.104` | `0.14` | 前腿约 `0.80`，后腿约 `0.75` |
| Disk robot train | `0.125` | `0.125` | `0.24` | 约 `0.52` |
| Disk robot non-train | `0.110` | `0.110` | `0.24` | 约 `0.46` |

这说明 disk robot 的髋关节相对它更大的圆盘来说更靠中心。绝对距离上，训练版的髋关节在 x 方向略微更分开；但按躯干尺寸归一化后，它们明显更收在圆盘下面。

## 腿长与零位足端 reach

| 参数 | Pupper disk visual | Disk robot train |
|---|---:|---:|
| 第一段有效 offset，hip 到 knee/body-3 原点 | `0.08445` | `0.16000` |
| 远端 offset，knee/body-3 原点到 foot site/foot geom | `0.08984` | `0.14000` |
| 建模 offset 总和 | `0.17429` | `0.30000` |
| 零位右前腿 hip-to-foot 向量 | `(-0.00619, -0.01800, -0.11156)` | `(0.00000, 0.00000, -0.30000)` |
| 零位 hip-to-foot 向量长度 | `0.11317` | `0.30000` |

Pupper 的腿是 XML 中带空间折叠结构的 CAD-derived linkage。它的局部 offset 是：

```text
leg_*_3 body offset: 0 -0.0494 0.0685
foot site offset:    0.06231 +/-0.06216 0.018
```

应用 body quaternion 后，零位足端主要位于髋关节下方，但在 x/y 方向也有少量偏移。相比之下，disk robot 的腿被刻意简化成了很直接的竖直链：

```text
upper capsule: fromto 0 0 0  0 0 -0.16
lower capsule: fromto 0 0 0  0 0 -0.14
foot sphere:   pos 0 0 -0.14
```

因此，disk robot 在零位下是直立竖直腿链。它的 hip-to-foot 零位长度约为 Pupper 零位 hip-to-foot 长度的 `2.65x`，约为 Pupper 建模 offset 总和的 `1.72x`。

## 脚和接触几何

| 参数 | Pupper disk visual | Disk robot train |
|---|---:|---:|
| 脚碰撞类型 | sphere | sphere |
| 脚球半径 | `0.01995` | `0.035` |
| 脚摩擦 | 继承 collision 默认值 `0.8 0.02 0.01` | foot class: `1.4 0.04 0.002` |
| 接触维度 | 默认 `condim=3` | foot class `condim=4` |

disk robot 的脚明显更大：半径 `0.035 m` 对比 `0.01995 m`，约为 Pupper 的 `1.75x`。它还使用了更高的主摩擦系数。

## 关节和执行器差异

| 参数 | Pupper disk visual | Disk robot train |
|---|---|---|
| 关节命名 | `leg_front_r_1`, `leg_front_r_2`, ... | `fl_hip_abd`, `fl_hip_flex`, `fl_knee`, ... |
| 关节 damping | `0.01` | `0.06` |
| 关节 armature | `0.0016` | `0.002` |
| 关节轴 | 旋转 body frame 中的局部 `0 0 1` | abduction 是 `1 0 0`，hip/knee flex 是 `0 1 0` |
| actuator 类型 | `general` actuators | `position` actuators |
| disk robot 位置控制增益 | 无 | hip abd `kp=55`，hip flex `kp=70`，knee `kp=55` |
| disk robot 力限制 | 无 | 所有 actuator 都是 `-8 8` |

Pupper 的关节轴依赖旋转后的 body frame 才有实际意义。disk robot 的关节轴则直接在躯干坐标里可读，更容易理解和调试，但也更不像原始 CAD 腿结构。

代表性关节范围：

| 关节组 | Pupper 范围 | Disk robot train 范围 |
|---|---|---|
| 第一关节 / 类 abduction | 右前 `-1.22 2.51`，左前 `-2.51 1.22`；后腿镜像 | 所有 hip abduction 都是 `-0.9 0.9` |
| 第二关节 / 类 hip flexion | 右侧 `-0.42 3.14`，左侧 `-3.14 0.42` | 所有 hip flexion 都是 `-3.1 3.1` |
| 第三关节 / 类 knee | 右侧 `-2.79 0.71`，左侧 `-0.71 2.79` | 所有 knee 都是 `-2.4 2.4` |

## 质量和惯量建模

Pupper disk visual 使用显式 inertial：

| 部件 | 质量 |
|---|---:|
| base body | `1.506` |
| 每条腿第一段 | `0.18` |
| 每条腿第二段 | `0.186` |
| 每条腿第三段 | `0.05` |
| 显式质量近似总和 | `3.17` |

Disk robot train 使用基于 density 的 primitive geometry：

| class | 几何 | density |
|---|---|---:|
| `disk` | cylinder，半径 `0.24`，半高 `0.045` | `450` |
| `leg` | capsule，半径 `0.018` | `350` |
| `foot` | sphere，半径 `0.035` | `500` |

忽略很小的 anchor marker geom 后，按 density 粗略估算的质量：

| 部件 | 近似质量 |
|---|---:|
| 躯干圆盘 cylinder | `~7.33` |
| 一条 upper capsule，长度 `0.16` | `~0.068` |
| 一条 lower capsule，长度 `0.14` | `~0.061` |
| 一个 foot sphere | `~0.090` |
| 总质量近似 | `~8.21` |

这只是根据 XML primitive 尺寸和 density 做的解析估算；MuJoCo 编译后的实际质量可能会因为 capsule 几何定义略有差异。关键的定性结论是：与 Pupper-style 文件相比，disk robot XML 更重，而且质量更由圆盘躯干主导。

## Keyframe 姿态差异

| Keyframe | Pupper disk visual | Disk robot train |
|---|---|---|
| home/stand root z | `0.28` | `stand=0.408`, `walk_stand=0.408` |
| folded root z | `0.24` | `0.24` |
| qpos 数量 | `19` | `19` |
| ctrl 数量 | `12` | `12` |

两者都是带 free root 的 12 actuator 四足机器人，但站立高度非常不同。disk robot 的 standing keyframe 把 root 放得更高，这与它更长的竖直腿一致。

## 对训练的含义

这些 XML 差异解释了为什么 Pupper-style 模型上的行为不能直接迁移到当前 disk robot：

1. **disk robot 的躯干杠杆问题更强。** 它的圆盘半径更大，但髋关节相对圆盘更靠中心。圆盘触地、躯干 pitch/roll 会更主导行为。
2. **disk robot 的腿更长也更简单。** 它有更大的竖直 workspace，但没有继承 Pupper 的折叠 CAD linkage 几何。
3. **disk robot 几何更 torso-heavy。** 粗略质量估计中，圆盘躯干占主导；速度跟踪奖励之外需要保留适度的躯干姿态、圆盘触地和跌倒约束，但不应指定固定接触时序。
4. **Pupper 的零位腿已经是空间折叠结构。** disk robot 的零位是直链竖直腿。因此两者必须分别标定 `q_stand` 和每关节 action scale，不能直接复制关节残差数值。
5. **训练 XML 把 disk hip 在 x 方向向外移了。** `disk_quadruped_extreme_train.xml` 使用 `x=+/-0.10`，而 `disk_quadruped_extreme.xml` 使用 `x=+/-0.08`。这把前后髋间距从 `0.16 m` 增加到 `0.20 m`。

后续调参最重要的几何旋钮可能是：

- 圆盘半径和质量/density；
- 髋关节锚点相对圆盘中心的 z offset；
- 前后髋关节间距；
- 脚半径和摩擦；
- 腿是否继续保留简化竖直直链，还是向 Pupper 的折叠 linkage 靠近。
