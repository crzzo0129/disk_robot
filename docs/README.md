# Disk Robot 文档索引

本文档目录按“当前规范”和“模型背景”分层。训练实现发生冲突时，以当前规范为准，历史说明不能覆盖它。

## 当前规范

- [omnidirectional_training_pipeline.md](omnidirectional_training_pipeline.md)：行走控制与训练的唯一依据。最终策略不使用运行时开环 gait。
- [common_commands.md](common_commands.md)：当前可用的检查、仿真和训练冒烟命令。
- [todo_extreme_disk_quadruped.md](todo_extreme_disk_quadruped.md)：按当前 pipeline 排列的实施任务。

## 模型背景

- [design_extreme_disk_quadruped.md](design_extreme_disk_quadruped.md)：极端圆盘机身机器狗的机械与 MJCF 建模原则。
- [pupper_vs_disk_robot_xml_geometry.md](pupper_vs_disk_robot_xml_geometry.md)：Pupper 与 disk_robot 的 XML、尺寸、碰撞和惯量对比。

## 约定

- 旧的 `Open-loop Gait + Residual RL` 路线已停止作为训练主线。
- 开环轨迹仍可离线生成示范数据，但不能作为最终策略的运行时动作锚点。
- `walk_smoke.py` 和 MJX 环境均使用静态站姿位置残差；`gait.py` 仅保留为离线 teacher 工具。
- 每次改变动作、观测、奖励或 command 定义时，先更新训练 pipeline，再更新实现与测试。
