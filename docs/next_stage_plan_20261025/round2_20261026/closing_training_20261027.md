# 六次消融训练：实时进度与完成记录

本次只训练G0、Q-global，各seed42/43/44，共6次、600轮；不依据seed42筛掉控制。
协议见[冻结设计](closing_protocol_20261027.json)。原16次结果保持只读。
所有指标均为开发验证表现，不能作为独立测试结论。

<!-- CLOSING_TRAINING_STATUS_BEGIN -->

状态：**complete**；更新时间：2026-10-27T22:58:15-04:00。

已完成 **6/6次**，合计 **600/600轮**。

| 模型 | seed | 状态 | 已完成轮次 | 当前最佳开发F1 |
|---|---:|---|---:|---:|
| G0 | 42 | complete | 100/100 | 0.8261 |
| Qglobal | 42 | complete | 100/100 | 0.8269 |
| G0 | 43 | complete | 100/100 | 0.8209 |
| Qglobal | 43 | complete | 100/100 | 0.8209 |
| G0 | 44 | complete | 100/100 | 0.8283 |
| Qglobal | 44 | complete | 100/100 | 0.8166 |

六次训练及最终验收全部完成。结论与配对比较见输出目录`完成汇报.md`，均为开发验证结果。

<!-- CLOSING_TRAINING_STATUS_END -->

本文件在后台运行期间每15秒自动刷新。日期和预计时间采用America/New_York。
模型每轮保存进度和恢复权重，每5轮保存快照并按固定31阈值评价。

机器可读进度：`outputs/next_stage_v3_closing_controls/controller_status.json`。
完整输出：`outputs/next_stage_v3_closing_controls/controller.log`。
每次训练：`outputs/next_stage_v3_closing_controls/01_screen/<model>/seed_<seed>/`。
全部完成后，本文件自动写入验收结果和配对比较，并生成输出目录中的`完成汇报.md`。

若异常停止，本文件会显示错误，完整追踪保存在输出目录；不自动换参数或额外增加trial。
通过新建输出目录内的`STOP_REQUESTED`文件，可要求在当前trial结束后停止队列。
