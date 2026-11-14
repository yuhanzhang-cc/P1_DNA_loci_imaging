# A–E骨架验收与明早交接

13/13 dry-run通过；22项单测通过；零GPU/优化器更新/新baseline/独立test图像读取。362个受保护文件哈希未变，正式训练计数仍41，既有结论不变。

S0/P0适配器和正式test实时数据/推理接口仍未实现。dry-run通过不表示可训练或已做独立测试。

完整清单：[FILES_BY_GROUP.csv](FILES_BY_GROUP.csv)；每项一行复跑命令及口径：[README.md](README.md)。

|项|新增/修改文件与示例路径|证据摘要|日志|
|---|---|---|---|
|A1|[train_mechanism.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/train_mechanism.py); [S0.json](configs/S0.json); [P0.json](configs/P0.json)|schema_valid=true; launch_ready=false; adapter_implemented=false; optimizer_updates=0|[A1.log](runs/20261103T194927Z/evidence/A1.log)|
|A2|[capacity_control.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/capacity_control.py); [CAP_P0.json](configs/CAP_P0.json); [CAP_P0.json](runs/20261103T194927Z/examples/CAP_P0.json)|expected671719; measured_new_adapter=false; alias S0|[A2.log](runs/20261103T194927Z/evidence/A2.log)|
|A3|[status.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/status.py); [04_status.sh](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/04_status.sh); [registry.json](configs/registry.json)|S0/P0/CAP_P0三种子not_started; 完成计数41|[A3.log](runs/20261103T194927Z/evidence/A3.log)|
|B4|[ablation_matrix.json](configs/ablation_matrix.json)|非法minus_signed排除; base复用; minus_support别名去重|[B4.log](runs/20261103T194927Z/evidence/B4.log)|
|B5|[ablation_runner.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/ablation_runner.py)|6个命令仅打印; commands_executed=0|[B5.log](runs/20261103T194927Z/evidence/B5.log)|
|B6|[aggregate_results.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/aggregate_results.py); [per_seed_metrics.csv](runs/20261103T194927Z/examples/aggregate_existing/per_seed_metrics.csv); [model_summary.csv](runs/20261103T194927Z/examples/aggregate_existing/model_summary.csv); [schema.json](runs/20261103T194927Z/examples/aggregate_existing/schema.json)|18行现有结果/6组摘要; 60个统计复现;核心容差1e-12/定位沿用2e-7|[B6.log](runs/20261103T194927Z/evidence/B6.log)|
|C7|[formal_test_eval.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/formal_test_eval.py); [validation_model_locks.json](configs/validation_model_locks.json); [independent_test.json](configs/independent_test.json)|9个validation锁核验; test图像读取0; test_evaluated=false|[C7.log](runs/20261103T194927Z/evidence/C7.log)|
|C8|[ranked_ap.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/ranked_ap.py); [test_scaffolds.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/test_scaffolds.py)|sklearn1.8.0合成参考PASS;10 passed,12 deselected|[C8.log](runs/20261103T194927Z/evidence/C8.log)|
|C9|[weak_policy.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/weak_policy.py); [weak_policy.json](configs/weak_policy.json)|372点/96弱点;旧标签逐项相同; refitted=false|[C9.log](runs/20261103T194927Z/evidence/C9.log)|
|D10|[reproduce.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/reproduce.py); [requirements-scaffold-qa.txt](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/requirements-scaffold-qa.txt); [scaffold_common.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/scaffold_common.py); [bootstrap_scaffold.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/bootstrap_scaffold.py); [run_scaffold_checks.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/run_scaffold_checks.py); [write_handoff.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/write_handoff.py); [model_summary.csv](runs/20261103T194927Z/examples/clean_reproduction_table/model_summary.csv)|打印干净checkout步骤;只读表复现通过; clean_checkout_executed=false|[D10.log](runs/20261103T194927Z/evidence/D10.log)|
|D11|[README.md](../../../../../README.md); [experiment_registry.csv](../../../experiment_registry.csv); [inspect_docs.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/inspect_docs.py)|三项not_started/指标空白;正式训练计数41|[D11.log](runs/20261103T194927Z/evidence/D11.log)|
|E12|[blind_audit.py](../../../../../deepblink_pytorch/scripts/diagnostics/scaffolds/blind_audit.py); [blind_audit_response_template.csv](templates/blind_audit_response_template.csv); [blind_audit_column_schema.json](templates/blind_audit_column_schema.json)|60ID;映射无泄漏;responses_complete=false;240必填字段仍缺|[E12.log](runs/20261103T194927Z/evidence/E12.log)|
|E13|[synthetic_key.csv](runs/20261103T194927Z/examples/synthetic_review/synthetic_key.csv); [synthetic_reviewer_1.csv](runs/20261103T194927Z/examples/synthetic_review/synthetic_reviewer_1.csv); [synthetic_reviewer_2.csv](runs/20261103T194927Z/examples/synthetic_review/synthetic_reviewer_2.csv); [agreement_by_stratum.csv](runs/20261103T194927Z/examples/synthetic_review/agreement_by_stratum.csv); [automatic_proxy_crosstab.csv](runs/20261103T194927Z/examples/synthetic_review/automatic_proxy_crosstab.csv); [provenance.json](runs/20261103T194927Z/examples/synthetic_review/provenance.json)|两份SYNTHETIC审阅者/60个SIM编号;真实盲审未完成|[E13.log](runs/20261103T194927Z/evidence/E13.log)|

完整命令/状态：[checks.json](runs/20261103T194927Z/checks.json)；单测：[all_unit_tests.log](runs/20261103T194927Z/evidence/all_unit_tests.log)；B6逐值复现：[B6_gold_reproduction.json](runs/20261103T194927Z/evidence/B6_gold_reproduction.json)。

## 明早审查顺序

1. A3 [registry.json](configs/registry.json)：CAP_P0=S0别名，不能重复计数；新04_status包装器不修改封存脚本。
2. B4 [ablation_matrix.json](configs/ablation_matrix.json)：6次未来独立训练预算、合法去除和不可表示的非法组合。
3. C9 [weak_policy.json](configs/weak_policy.json)：固定训练期阈值的严格小于边界、原图局部SD与unclassifiable处理。
4. E12 [blind_audit_response_template.csv](templates/blind_audit_response_template.csv)：先盲审填写；TP/FP裁定仅在解盲结合标注后填写。空白表不算审完。

## 留TODO的事项

- 待blind_audit/机制确认：P0邻域半径（候选2，正式值null）、归属规则、loss归一化、是否启动S0/P0、正式止损/继续规则。
- 待另行实现：有符号offset/邻域监督适配器、真实参数计数和forward/backward预检；本轮不改网络或round2_protocol.py。
- 待独立数据/协议：新test manifest、独立性和排重审查、完整候选导出、实时推理适配器与正式test执行。
- 待真实人工：60例评分、第二审阅者、分歧裁定和解盲TP/FP；E13假数据不提供真实一致率结论。

## 一键复跑

python deepblink_pytorch/scripts/diagnostics/scaffolds/run_scaffold_checks.py --data-root D:\Projects\P1_DNA_loci_imaging --dry-run

每次另建UTC目录，不覆盖旧结果。私有结果需显式挂载；本轮未做新的git checkout或安装依赖，也未提交/推送，以便先审VS Code diff。QA requirements只定义可选隔离环境。
