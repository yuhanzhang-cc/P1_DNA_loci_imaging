# 2026-11-11：审批后开发研究完整收尾准备

这是本轮最新总入口。真实审批已接收，启动直接看 **[AUTODL_START.md](AUTODL_START.md)**；实测证据、最终上传包和限制以 **[REPORT.md](REPORT.md)**、`CHECKS.json`及`RELEASE.json`为准。[ONE_DAY_RUNBOOK.md](ONE_DAY_RUNBOOK.md)保留审批前的完整分支设计，旧r2包命令已由最终已审批包替代。

源码：`deepblink_pytorch/scripts/diagnostics/completion_v1/`。既有代码只修改四个文件：`handoff_v2/decision.html`、`freeze.py`、`pipeline.py`和`mechanism_v1/contracts.py`，分别增加显式审批选项、校验/继续策略和新源码哈希绑定。`round2_protocol.py`、网络结构、训练配方和历史baseline结果未改。

## 已预制的完整链

真实图像审阅 → 一次研究决定 → 配置冻结 → 实际CUDA/初始化预检 → 既定S0/P0六次及可选scratch三次 → 统一阈值指标/完整AP/共同TP误差 → 描述统计与预声明对比 → 五张开发图（SVG/PDF/PNG及source tables）→ 英文Methods/Results/Discussion等草稿 → 同设备推理计时 → 冻结模型评价包 → 一个总返回ZIP → 本地SHA/CRC/模型预算验收。

有真实独立数据时：审查采集身份 → 自动文件哈希/manifest → 冻结模型独立读出 → 生物学组区间及第六张图 → 回填英文结果。没有数据时保留明确PENDING，不能把旧内部测试或合成数据当独立验证。

## 重点审阅

1. 新研究决定表的 `full_registered_six` 和scratch选项是否符合你的计划；两者默认关闭，需要真实选择。
2. [ablation_matrix.json](ablation_matrix.json)：参数对照S0被复用，最多六次核心训练；不是完整因子消融。
3. [NOVELTY_AND_CLAIMS.md](NOVELTY_AND_CLAIMS.md)：S0/P0测试监督机制，未凭空建立新架构或“我们最好”。
4. `manuscript_templates/` 和生成的英文draft：数值绑定实际表，采集细节/审查/待执行实验保留空位。

## 一行复跑

以下均在项目根目录运行，用已有CPU环境；output换成全新目录，不覆盖本次证据：

```powershell
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py audit --root . --output "<新审计目录>"
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py finish --root . --output "<新开发报告目录>"
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py build --root . --output "<新私有上传包目录>"
```

各环节的额外命令与证据在`ARTIFACTS.csv`；环境沿用已跟踪的requirements，未安装/升级共享环境。使用的科学图件和统计技能引用已记录在[SOFTWARE_ASSISTANCE.md](SOFTWARE_ASSISTANCE.md)，元数据从主要来源核对。通用图件尚未按指定期刊做终版验收。
