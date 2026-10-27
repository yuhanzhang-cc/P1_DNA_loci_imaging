# R1/S1种子补齐：独立追加预算与执行日志

授权：用户在2026-10-26的今日汇报后同意补齐训练，并要求全部完成后更新日志。分支保持`codex/clean220-round2`。

<!-- DETAIL_COMPLETION_STATUS_BEGIN -->

状态：**全部完成并通过验收**；完成时间2026-10-27T03:32:30-04:00（America/New_York）。新增4次/400轮，原12次结果保持不变。

R1-R0平均ΔF1 +0.0136，3/3种子为正；R1-S1平均ΔF1 +0.0020，2/3种子为正。

R1在本次开发复验中相对R0和S1均呈正向平均差，但仍须核对弱点约束、固定FP预算和独立验证；不能据此认定架构创新或独立泛化收益。

结果见[完成汇报](../../../outputs/next_stage_v3_detail_completion/完成汇报.md)及[最终验收](../../../outputs/next_stage_v3_detail_completion/delivery_checks.json)。

<!-- DETAIL_COMPLETION_STATUS_END -->

## 固定范围

| 新任务顺序 | 模型 | seed | epoch |
|---|---|---:|---:|
| 1 | R1 细节选择 | 43 | 100 |
| 2 | S1 普通细节投影 | 43 | 100 |
| 3 | R1 细节选择 | 44 | 100 |
| 4 | S1 普通细节投影 | 44 | 100 |

追加上限为4次训练、400轮。此前v3的12次/1200轮及原锁不修改，两部分合计16次。原R0三个种子和R1/S1的seed42直接复用；没有新架构、学习率、loss、增强、组合或阈值搜索，也不按结果追加任务。本次补齐由前次结果启发，仍为开发研究，不是新的盲法验证。

训练函数直接复用字节未变的`optimize_v3.run_trial`，独立入口[`complete_detail_v3.py`](../../../deepblink_pytorch/scripts/complete_detail_v3.py)只指定新授权锁及输出目录。固定176训练/22验证、batch8、Adam lr0.001、wd0、dropout0.3、XPU bfloat16、channels-last、4线程、无增强；Dice+2RMSE和原31阈值×20个checkpoint选模不变。原测试数组不读取。

训练源码的52个文件必须与原R0登记逐项相同；每个新增trial必须与对应seed的R0保持相同配置及100轮样本顺序。全部新增任务都必须完成，之后同时报告R1−R0与R1−S1的三个配对差、均值/样本SD、正向种子数、弱点检出及相同FP预算的工作点，不选最好种子代替整体结论。

## 预检、保存与完成条件

- 启动前核对原12次已完成、源文件、原权重及指标哈希、440个原始文件、输入NPZ和固定split；保持原结果目录只读使用。PyTorch版本与XPU设备须匹配原对照。
- 新协议、辅助脚本快照与数据预检保存在`outputs/next_stage_v3_detail_completion/00_protocol/`。训练环境不新增依赖；日志时区从带哈希的America/New_York TZif文件读取。
- 每轮损失和恢复状态、每5轮权重/候选/阈值表、最终权重和曲线保存到`01_screen/<模型>/seed_<seed>/`。单worker文件锁阻止重复并发；动态状态每15秒更新。
- 只有四项均完成100轮/2200次更新、20×31工作点、参数及候选有限、配对顺序一致、原数据及受保护原结果未变后，才生成最终完成日志。任何失败保留现场并标记`needs_investigation`，不修改参数或自动重试算法错误。
- 同配置中断恢复须先勘察，再显式使用`--resume`；不会静默覆盖部分训练。`STOP_REQUESTED`在trial边界检查，立即停止需要确认并停止本worker的实际Python进程。

## 输出与复现

```text
outputs/next_stage_v3_detail_completion/
  00_protocol/lock.json
  00_protocol/data_audit.json
  controller_status.json
  execution_events.jsonl
  controller.log
  01_screen/{R1,S1}/seed_{43,44}/
  validation_comparison.csv           # 合并读取原12次与新增4次
  paired_summary.json                # R1-R0及R1-S1
  02_analysis/                       # 逐点/尺寸/FP预算/旧权重参照
  delivery_checks.json               # 全部完成后的验收
  完成汇报.md
```

在仓库根目录，用现有`.venv/clean220_xpu/Scripts/python.exe`运行`deepblink_pytorch/scripts/complete_detail_v3.py --prepare-only`登记，`--check-only`只核对，`--run`执行固定四项。实际任务启动后不要重复启动；文件锁只防并发，不替代对恢复现场的审计。

前次[2026-10-26今日汇报](daily_report_20261026.md)保留为原12次结果快照，不覆盖其历史结论。新入口、分析脚本和日志是本次VS Code diff的重点；核心训练/模型/数据代码与环境定义保持原样。
