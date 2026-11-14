# Review-to-report prefabrication v2 · 2026-11-11

最新工作总览：[OVERNIGHT_PREFAB_REPORT.md](OVERNIGHT_PREFAB_REPORT.md)。明早操作：[START_HERE.md](START_HERE.md)。机器验收：[CHECKS.json](CHECKS.json)。本轮**工程预制完成、真实人工回填/CUDA/正式训练未完成**，不是新增性能结论。

|产出族|输入 → 输出|一行复跑命令（项目根目录，`python` 指现有 CPU 环境）|证据|
|---|---|---|---|
|图像审阅表单|旧匿名 PNG/空表 → 新离线页，保留原显示规则|`python deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py reviewer --public docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/overnight_ready/blind_audit_public --output outputs/reviewer_fresh`|`evidence/javascript_form_contract.txt`；视觉点选未验收|
|真实回答校验|评分 CSV + 60 个匿名 ID → 完整性、来源检查|`python deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py check-review --responses responses.csv --expected-ids docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/implementation_20261104/configs/blind_audit_expected_ids.json`|新交接 34 单测；真正冻结还检查图像包 ID|
|方案冻结|真实 CSV + 批准决定 → 冻结配置/批准/来源哈希|`python deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py freeze --responses responses.csv --decision research_decision.json --config-dir outputs/next_stage_v4_recipe_20261101/19_prefabrication_20261111/p1_review_pipeline_20261111/configs --output outputs/next_stage_v4_recipe_20261101/19_prefabrication_20261111/decisions_fresh`|`evidence/synthetic_roundtrip.json`；合成回填被正式入口拒绝|
|后端取证|已有环境 → 环境/三种子初始化证据|`python deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py doctor --device cpu --output outputs/doctor_fresh`|`evidence/doctor_cpu.json`；CPU 通过不等于 CUDA 通过|
|队列/消融/容量接力|冻结方案 → seed42 筛查、条件四次、打包|`python deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py run --root outputs/p1_review_pipeline_fresh --dry-run`|`evidence/final_cloud_plan.txt`；最多六次，CAP_P0 为 S0 别名|
|云端上传包|只含开发集的旧源 → 独立密封包|`python deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py build --data-root . --output outputs/p1_review_pipeline_fresh`|`evidence/build_cloud_final.txt`；119 成员；CRC/SHA256 与源绑定已验|
|无训练基准表复现|现有 R0/H0/G0 九个导出 → 表/计划差异/报告|`python deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py report --data-root . --output outputs/reference_report_fresh`|`evidence/baseline_reproduction/REPORT.md`；六项未跑留空|
|回传验收和自动报告|返回 tar.gz/校验值/原上传 manifest → 新目录/报告|`ingest` 与 `report` 完整命令见 [START_HERE](START_HERE.md)|`synthetic_roundtrip.json`；合成包普通入口拒绝、显式工程通道往返成功|
|预制原图纠标材料|仅 owner key/原始验证数组 → 19 张原图/点/编号候选宏|`python deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py prepare-corrections --source-key docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/overnight_ready/private/blind_audit_source_key.csv --data-root . --output outputs/owner_materials_fresh`|`evidence/correction_materials.json`；原像素不变；盲审后才给审查者|
|ImageJ 核心 I/O|现有 Fiji/JDK + 合成图 → 亚像素/像素保护/CSV 检查|`python deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py fiji-qa --fiji-root C:/Users/Yuhan_Zhang/Downloads/fiji-stable-win64-jdk/Fiji.app --output outputs/imagej_core_fresh`|`evidence/imagej_core_qa.json`；不是交互拖点验收|
|标签修订分支|实际图上编辑的完整点表 → 独立版本提案/固定阈值诊断复评|`check-revision` / `propose-labels` / `recheck-labels` 命令见 [START_HERE](START_HERE.md)|`evidence/unit_handoff_final_34.txt`；未修改真实标签，不进入当前正式协议|

所有 `fresh` 目录应使用新的名字；不删除或覆盖旧输出。复跑需要已有的私有输入，它们不在 Git 中。新代码没有增加共享环境依赖：基本运行复用 `requirements-mechanism.txt` 中已验证版本；Node 仅用于 JS 合约测试，Fiji/JDK 仅用于可选本机宏 QA，AutoDL 主队列不需要两者。

## 复现本地检查

```powershell
python -m pytest -q deepblink_pytorch/scripts/diagnostics/handoff_v2/test_handoff.py
node deepblink_pytorch/scripts/diagnostics/handoff_v2/test_reviewer.js outputs/SYNTHETIC_QA_export.csv
```

当前 `.venv/spotiflow_cpu` 没安装 pytest，本轮复用既有基础环境中的纯 Python pytest 工具，导入后移除其 site-packages 路径，避免 Python 3.13 的可选二进制包污染 Python 3.12 Torch；精确命令见日志。没有安装/升级共享依赖。

证据分组：既有机制/诊断 37 + 骨架 22 + 新交接 34 = **93 个 Python 测试**。旧组合日志 `unit_cpu_68.txt` 包含 37 个既有和最初 31 个新交接；新标签与事务冻结检查加入后的最终新交接日志为 `unit_handoff_final_34.txt`，不将重复运行累加为新实验。

真实研究/隐私边界：`private/`、`qa_private/`、上传 ZIP、逐点返回数据不提交；公开的是代码、汇总、配置模板与检查。旧包/旧结果/原始标注保持原样，正式训练计数仍 41。实际 GPU 队列、完整 100 轮恢复和回传会以真正运行证据更新，今晚不伪造。
