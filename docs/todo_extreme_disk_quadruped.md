# 极端圆盘躯干四足机器人 TODO 与进度记录

## 当前状态

- 已确定第一版形态：正常四足布局 + 侧视近圆盘、前视窄厚度的 disk 型躯干。
- 第一阶段使用抽象 MJCF，不直接修改 Pupper v3。
- 已建立模型文件：`assets/disk_quadruped_extreme.xml`。
- 已保留 12 个单独 position actuator，不在 XML 中做前后腿物理耦合。
- 已人工标定两个暂定标准姿态：
  - `stand`：标准站立姿态。
  - `folded`：收腿/滚动准备姿态。
- 已添加工具脚本：
  - `scripts/view_extreme_disk_pose.py`：查看 `stand` / `folded`。
  - `scripts/control_extreme_disk_flex.py`：用键盘联动控制前腿、后腿或全部 `hip_flex`。
  - `scripts/interpolate_extreme_disk_pose.py`：播放 `stand <-> folded` 姿态插值。
  - `scripts/diagnose_extreme_disk_pose.py`：输出 torso 高度、接触数量、脚端高度、关节角和 ctrl。

## 已讨论的重要观点

- 当前模型不是球形机器人，而是正常四足机器人加极端圆盘躯干。
- 当前 XML 中的 `stand` 和 `folded` 由人工标定，后续脚本和训练优先引用这两个 keyframe。
- Viewer 中的网格显示是 MuJoCo 可视化选项，不是 PNG 警告导致，也不是 XML 损坏。
- `body pos` 是相对于父 body 的局部坐标；父 body 旋转会影响子 body 坐标系。
- `ctrlrange` 是 actuator 输入范围，不等于关节物理范围；关节范围由 joint 的 `range` / `limited` 决定。
- 探索阶段保留 12 个单独 actuator；前后腿 flex 联动放在 Python 控制脚本层实现，不改 XML actuator 结构。
- 姿态插值脚本只用于观察几何路径，不代表真实动力学控制已经能完成收腿。

## 第一阶段诊断结果

诊断命令：

```powershell
python -m scripts.diagnose_extreme_disk_pose --keyframe all
```

当前结果摘要：

- `stand`：
  - torso 中心高度 `z=0.3200`。
  - 第一帧有 4 个接触：四个脚都与地面接触。
  - 四个脚球中心高度约 `center_z=-0.0530`，脚球半径为 `0.035`，所以 clearance 约 `-0.0880`。
  - 这说明当前站立 keyframe 是“带穿入的接触初值”。MuJoCo 继续运行后会由接触求解器修正，但如果用 passive 卡在第一帧，会看到脚在地面以下。
- `folded`：
  - torso 中心高度 `z=0.2400`。
  - 第一帧有 6 个接触：躯干与地面接触，并且四个脚与躯干接触。
  - 四个脚已经收高，脚球 clearance 约 `0.1049`。
  - 这适合作为滚动准备姿态的第一版，但脚-躯干内部接触需要在后续训练/碰撞组设计中继续关注。

## 收腿与滚动切换策略假设

因为躯干是 disk 型、腹部是弧形，前后腿同时收起到一定程度时可能导致机器人不稳定。后续从 `stand` 到 `folded` 的切换不应只有简单同步插值。

建议策略：

- 初期允许前后腿同步向 `folded` 靠近。
- 当 torso 中心高度下降到阈值附近，例如 `z <= 0.25`，切换为前后腿分阶段收起。
- 向前滚动时，先收前腿，让机器人借助重心偏移和圆盘腹部接触向前滚起来。
- 前腿完成一定折叠进度后，再收后腿，进入完整 `folded` 姿态。
- 后续可以把该过程做成 scripted reference trajectory，再用于 curriculum 或 imitation-style reward。

## TODO

### 阶段 1：模型与姿态工具

- [x] 建立 `disk_quadruped_extreme.xml`。
- [x] 保留 12 个单独 actuator。
- [x] 标定 `stand` keyframe。
- [x] 标定 `folded` keyframe。
- [x] 添加姿态查看脚本。
- [x] 添加前后腿 flex 键盘联动脚本。
- [x] 添加 `stand <-> folded` 姿态插值脚本。
- [x] 添加姿态诊断脚本：输出脚端高度、接触数量、torso 高度、关节角和 ctrl。
- [x] 检查当前 `stand` 第一帧接触状态，并记录脚球穿入现象。
- [x] 检查当前 `folded` 第一帧接触状态，并记录躯干接地和脚-躯干接触。

阶段 1 当前可以视为完成。进入训练前，如果要提高物理干净程度，可以再单独做一次“碰撞/初始高度修正”小迭代；但这不阻塞我们把第一版模型冻结下来作为实验基线。

### 阶段 2：行走训练准备

- [ ] 为 disk quadruped 新建独立 MJX/Brax environment，不直接复用 curl environment。
- [ ] 定义 walk observation：base pose/velocity、joint qpos/qvel、foot contact、command velocity。
- [ ] 定义 walk reward：前向速度、姿态稳定、能耗、足端接触、圆盘躯干不过早滚动。
- [ ] 做随机动作 smoke test。

### 阶段 3：滚动训练准备

- [ ] 从 `folded` keyframe 初始化 rolling task。
- [ ] 定义 rolling reward：前向位移、圆盘角速度、保持折叠、减少异常腿地面碰撞。
- [ ] 记录 disk 躯干接触状态和角速度。
- [ ] 验证人工扰动或简单脚本能让躯干产生滚动趋势。

### 阶段 4：切换训练准备

- [ ] 实现 `stand -> folded` scripted reference trajectory。
- [ ] 加入 torso 高度阈值，例如 `z <= 0.25` 后切换为分阶段收腿。
- [ ] 向前滚动默认顺序：先前腿，后后腿。
- [ ] 设计 `walk -> roll` 和 `roll -> walk` 的静止起步任务。
- [ ] 后续再扩展到动态行走中切换。
