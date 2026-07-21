# Disk Robot 文档索引

本文档目录按“当前可执行规范”“长期路线”和“模型背景”分层。训练实现发生冲突时，以当前可执行规范为准。

## 当前可执行规范

- [forward_teacher_student.md](forward_teacher_student.md)：当前对称 IK、privileged Teacher、phase-conditioned BC 和 Student-only 验收的运行依据。
- [student_imitation_failure_debugging.md](student_imitation_failure_debugging.md)：T3--T8 从低离线 MSE、闭环失败到定位 previous-action 高增益自反馈的完整证据链与通用调试方法。
- [common_commands.md](common_commands.md)：当前可用的检查、仿真和训练冒烟命令。
- [todo_extreme_disk_quadruped.md](todo_extreme_disk_quadruped.md)：按当前 pipeline 排列的实施任务。

## 长期路线

- [omnidirectional_training_pipeline.md](omnidirectional_training_pipeline.md)：从当前低速固定前进扩展到无运行时 gait 的全向控制。它记录方向，不覆盖当前命令和参数。

## 模型背景

- [design_extreme_disk_quadruped.md](design_extreme_disk_quadruped.md)：极端圆盘机身机器狗的机械与 MJCF 建模原则。
- [pupper_vs_disk_robot_xml_geometry.md](pupper_vs_disk_robot_xml_geometry.md)：Pupper 与 disk_robot 的 XML、尺寸、碰撞和惯量对比。
- [structure_variant_study.md](structure_variant_study.md)：腿长、髋距和圆盘半径的结构扫描、候选尺寸与 folded 包络。
- [structure_variant_study_zh.md](structure_variant_study_zh.md)：结构参数研究的中文版与复现步骤。
- [rolling_structure_com_study_zh.md](rolling_structure_com_study_zh.md)：结构、圆盘本体质心与整机质心对轴向滚动的联合扫描。

## 约定

- 固定 gait residual 只用于当前 Teacher 的探索引导，不是最终 Student 的运行时接口。
- 当前 T8 Student 从 `stand` 输出完整关节位置动作，不读取 IK、期望接触、足部接触或 previous action；它使用 controller-owned phase clock。
- 旧的 `mjx_train_walk` 从零 PPO 和 `teacher_blend` 退火实验不再是当前主线。
- 每次改变动作、观测、奖励或 command 定义时，先更新训练 pipeline，再更新实现与测试。
