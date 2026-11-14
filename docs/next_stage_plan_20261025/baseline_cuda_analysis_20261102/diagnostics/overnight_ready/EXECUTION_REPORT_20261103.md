# 过夜准备：端到端试跑结果

自动运行器已完成一次真实端到端试跑，UTC 2026-11-03 05:12:14–05:13:08，约54秒；本次没有训练或架构变更。

- 独立重放六个冻结模型，`peak_offset_diag.csv`与原表**字节一致**。
- 重新生成三层各20例的60例匿名包并验收。
- 60个定量裁剪与原始像素一致，60个统一显示映射一致，60个候选圈位置一致。
- 公开ZIP共123文件，不含模型、分层、来源映射或定量裁剪。
- 输入与训练代码哈希保持不变；优化器更新数0。
- 所有人工评分字段保持空白，人类盲审**仍未完成**。

原始执行记录：

`../overnight_runs/20261103T051214Z/status.json`

完整日志：

`../overnight_runs/20261103T051214Z/runner.log`

这次试跑已验证自动部分可以执行。下一次过夜重放会另建UTC时间子目录，不覆盖现有数据。此阶段不需要租GPU；自动部分结束后停止，真实下一输入是匿名人工评分表。S0/P0的六次拟议机制训练仍未启用，不能把本次试跑报告成新模型实验。

复跑命令：

```powershell
& D:\Projects\P1_DNA_loci_imaging\.venv\spotiflow_cpu\Scripts\python.exe D:\Projects\P1_DNA_loci_imaging\deepblink_pytorch\scripts\diagnostics\run_overnight_no_train.py --data-root D:\Projects\P1_DNA_loci_imaging --execute
```
