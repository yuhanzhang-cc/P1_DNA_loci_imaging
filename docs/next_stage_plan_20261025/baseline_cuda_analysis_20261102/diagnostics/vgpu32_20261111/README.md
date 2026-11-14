# vGPU32 独立交接包（2026-11-11）

最新版硬件替代入口：[AUTODL_START.md](AUTODL_START.md)。原5090包、真实审查、冻结配置和历史结果均保留；本页新增一个vGPU实验轨道，不替换原协议。

5090不是模型训练的必要条件；严格复现原5090环境时仍使用[原交接](../completion_20261111/AUTODL_START.md)。AutoDL把vGPU-32G列作RTX4080S 32G；实际名称、计算能力、显存及软件必须由服务器预检确认，不能只凭租赁页名称放行。[AutoDL](https://autodl.com/)、[官方GPU性能说明](https://www.autodl.com/docs/gpu_perf/)、[NVIDIA计算能力表](https://developer.nvidia.com/cuda/gpus)。

新包保留原9次训练：S0/P0各3 seed、Spotiflow scratch 3 seed。S0/P0配方、初始化哈希、100轮/2200 updates及原有恢复检查均未放宽；没有新增H0训练或独立test数据。新卡下P0–S0可作同硬件对照，S0–历史H0等对比须注明跨硬件。不同硬件即便确定性设置相同，也不保证逐位相同。[PyTorch数值精度说明](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)

新增源文件：`completion_v1/hardware_policy.py`、`transfer_check.py`、`test_hardware_policy.py`；相关构建器/doctor/运行器仅增加显式配置、前置检查、来源记录与报告限定语。不修改`round2_protocol.py`、网络或历史结果。

本地检查及封包证据见`CHECKS.json`、`RELEASE.json`和`evidence/`。本地通过不代表新服务器CUDA已经通过；失败必须保存日志，不得在看到结果后放宽初始化或0.005px门槛。
