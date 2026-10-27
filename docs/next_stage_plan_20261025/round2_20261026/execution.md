# 第二轮执行记录

分支：`codex/clean220-round2`；起点提交：`9edd529`。用户于2026-10-26授权按scope开展下一轮优化。

## 最终状态：12次训练全部完成

2026-10-26 22:35（America/New_York），控制器完成最终验收并正常退出：12次训练、1200轮、240个checkpoint，未做测试推理。R0/R2/S2三种子平均验证F1为0.8065/0.8103/0.8143；R2−R0平均仅+0.37个百分点，只有1/3种子为正，且低于简单控制S2，尚未建立稳定的核心架构收益。R1/S1各只有seed42，不作多种子结论。

详见[今日汇报](daily_report_20261026.md)、[最终验收](../../../outputs/next_stage_v3/delivery_checks.json)及[结果解释](../../../outputs/next_stage_v3/02_analysis/结果解释.md)。原12次预算已用满、原控制器已退出；此后用户另授权4次R1/S1种子补齐，独立状态见[补齐日志](detail_completion.md)。下文首批结果和启动说明作为过程记录保留，不应再运行原启动命令。

## 已完成的实施与预检

- 主干增加第一次解码和第二次编码的特征接口；旧权重在22张验证图的float32/NCHW输出与v2存档逐值一致，最大差异0。
- 实现R1/S1细节路径、H0密集heatmap/offset、R2/S2上下文路径；预测/评价入口按checkpoint类型处理新表示。新训练使用独立`optimize_v3.py`，不替换旧训练协议。
- 独立生成100轮样本顺序；所有模型在同seed使用相同顺序，额外模型初始化不影响采样。20个预定epoch快照各31阈值，主次选模规则冻结。
- 63项定向测试通过，包括开发数据边界、亚像素/整数/边缘坐标、padding、模型加载、梯度、简单控制及加速阈值匹配与原匹配计数的一致性。
- 440个原始文件哈希、原数据NPZ一致；开发集为176训练/22验证，训练4156点、验证372点。dense同像素冲突均为0；弱点真值定义沿用训练下四分位，验证96点。人工盲法标注复核仍待完成。
- 六模型均通过4张256＋4张512的XPU训练批次及float32推理检查。R0参数1,258,723；H0为671,719；R1/S1为1,265,350/1,265,336；R2/S2均1,275,623。前向初测均达到本轮工程目标；不能把微小计时差称作稳定速度收益。
- 10张预定训练图上的对角平移敏感性审计已保存，原图只归一化一次、共同ROI边距67px。结果有变化，但循环边界和全局SE使其不能单独建立网格相位的因果结论。

预检证据位于[`00_protocol`](../../../outputs/next_stage_v3/00_protocol/implementation_validation.json)，实际协议见[lock.json](../../../outputs/next_stage_v3/00_protocol/lock.json)。原YAML草案保留，不以改写旧文件掩盖协议变化。

## 首批训练

已启动R0→H0→R1→S1，各100轮、seed42、batch8、Adam lr0.001、wd0、无增强；训练bfloat16，点推理float32，均使用固定176/22开发数据。每轮记录损失和恢复状态，每5轮保存原始快照、未阈值化候选集及31个验证工作点。最低原生损失选模与点F1选模保留为同一批20权重上的配对分析。

日志：[`next_stage_v3_train_phase1.log`](../../../outputs/next_stage_v3_train_phase1.log)。逐trial配置、源码快照、实际样本顺序、训练状态与结果写入`outputs/next_stage_v3/01_screen/<ID>/seed_<seed>/`。完整trial才写入`validation_comparison.csv`；训练中最优值不是最终结果。所有搜索均不读取测试数组。

训练过程使用临时AC-only唤醒请求，退出或转为电池供电时释放，不修改全局电源策略。每轮恢复文件用于基础设施中断后的同配置恢复；部分运行不会静默从头覆盖。

### 首个完整结果与历史参照

R0已完成100轮。预定主规则选择epoch85、p=0.5：验证点F1 **0.804438**，TP/FP/FN为290/59/82，弱点Recall为41/96。相同20权重中，最低原生损失选择epoch100、p=0.005807：F1为0.801040，TP/FP/FN为308/89/64，弱点Recall为48/96。主规则只提高0.003398，且弱点Recall更低，不能将选模调整描述为弱点问题已解决。

旧220的seed42权重在本轮31阈值下重新评价其已存验证预测，仍为F1 **0.828229**、TP/FP/FN 311/68/61。其训练顺序和checkpoint选择机会与v3不同，不能取代新R0的配对对照；同时必须保留这个更高的历史参照，避免候选只超过一次较弱的新R0便被称为整体进步。此分析不做新模型推理、不读取测试数组，结果见[`inherited_weights_new_grid.csv`](../../../outputs/next_stage_v3/02_analysis/inherited_weights_new_grid.csv)。

首批过程快照中，H0曾在训练、R1/S1曾排队；现在均已完成。最终以各trial的`status.json`和`selection.json`为准。

## 自动续跑与验收

独立控制器[`continue_v3.py`](../../../deepblink_pytorch/scripts/continue_v3.py)按冻结条件完成了R2/S2及R0/R2/S2的seed43/44，最终正常退出。总计12次新训练，没有额外学习率、loss、分支宽度、增强或R1＋R2组合搜索。三项无训练检查验证了条件阶段、额外种子和第13次任务均会被拦截，见[`controller_guard_validation.json`](../../../outputs/next_stage_v3/00_protocol/controller_guard_validation.json)。下列启动与停止说明保留为复现记录，不代表当前仍在运行。

- 动态状态：[`controller_status.json`](../../../outputs/next_stage_v3/controller_status.json)，启动信息：[`controller_launch.json`](../../../outputs/next_stage_v3/controller_launch.json)。不要再启动第二个控制器或重复首批命令。
- 控制器日志：`outputs/next_stage_v3/controller.log`；后续阶段分别记录到`train_phase2.log`、`train_repeats.log`。报告和分析入口为`optimize_v3.py report`与`analyze_v3.py`。
- 结束后自动写出`阶段汇报.md`、`validation_comparison.csv`、`decision.json`，以及`02_analysis/`中的配对检测变化、原图尺寸分层、固定FP预算工作点和固定阈值的匹配半径敏感性。[结果解释.md](../../../outputs/next_stage_v3/02_analysis/结果解释.md)汇总新旧参照、完整/简单路线对照及下一步取舍；已在CPU上用首个完整R0结果运行通过，候选结果尚待生成。它们均为开发验证结果，不能当作独立泛化结论。
- 最终验收生成`delivery_checks.json`和`本轮完成记录.md`：检查100轮/2200次更新、20×31工作点、最佳权重与原始快照一致、参数及候选有限、同种子样本顺序一致、440个原始文件及NPZ哈希未变。
- 明确失败、非有限损失、断言异常会停止为`needs_investigation`并保留现场。仅当首批外部进程意外消失、源码未变且有完整epoch恢复文件时，才允许同配置恢复，不添加试验或改变参数。
- `outputs/next_stage_v3/STOP_REQUESTED`用于停止控制器及其后续调度。首批进程独立于控制器启动；若要立即停止仍在运行的首批，须同时确认并停止其实际Python进程，不能只依赖这个标记。实际PID和命令见启动记录/进程检查。

训练期间冻结核心源码和已登记协议；文档与独立分析脚本不改变训练数值路径。当前分支上的v3改动尚未提交，不修改已归档的v2结果。

VS Code重点：`unet.py`旧路径等价；R1的16位置索引和S1信息输入一致；H0的Gaussian标签/offset/局部峰解码；R2/S2中央采样排除及有效支持归一化；独立采样、620次选模上限、测试数据读取边界与所有失败记录。
