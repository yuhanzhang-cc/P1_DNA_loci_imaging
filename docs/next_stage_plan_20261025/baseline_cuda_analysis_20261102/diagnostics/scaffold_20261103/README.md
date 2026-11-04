# 无训练骨架交接：scaffold only, not run

日期：2026-11-03。G0仍为当前简单候选，既有结论与正式训练计数41不变。没有训练、GPU调用、网络结构改动、独立test数据读取或真实盲审评分。原`extracted_r0`、`extracted_h0`、`overnight_ready`逐文件保护。

**重要边界：dry-run通过只证明配置、脚本接口、合成计算和只读结果聚合可执行；不表示S0/P0适配器已经实现或可训练。** 两者仍为`not_started`，所有训练CLI强制要求`--dry-run`且没有非dry执行路径。没有为了测试创建优化器。正式test CLI同样只支持dry-run，数据加载/实时推理适配器留接口；完整ranked AP计算及有身份绑定的候选缓存评价函数已实现并通过合成测试。

## 明早审查顺序

1. **A3注册表**：`configs/registry.json`。S0/P0各三种子，CAP_P0是S0别名，不增加训练计数。新增状态包装器显示9条not_started记录；实际只规划6次独立未来训练。
2. **B4消融矩阵**：`configs/ablation_matrix.json`。H0复用；capacity=S0、full=P0、minus_support=S0别名；bounded-offset+neighborhood组合非法并被排除，不能为凑矩阵训练不可表示的目标。
3. **C9弱点定义**：`configs/weak_policy.json`。固定训练期阈值2.4566830772038957，严格小于才是weak。不是当前验证集信号四分位的Q1=2.4023403664，不能在test上重估阈值。
4. **E12回填模板**：`templates/blind_audit_response_template.csv`和`blind_audit_column_schema.json`。60个真实匿名ID，仅空白回填列，不含来源映射。模板结构可通过校验，但240个必填字段仍空缺，`responses_complete=false`；严格完成校验会拒绝它。

## 路径和复跑约定

从包含本轮代码的checkout根目录执行。`$R`指保存私有结果的目录，不能假定原始数据/权重已在Git中。表复现只依赖Python标准库；AP/弱点与单测使用NumPy，AP参考核对另外使用scikit-learn/pytest。本任务复用了现有CPU QA环境，未安装或改动共享训练环境。可选隔离QA依赖见新增`requirements-scaffold-qa.txt`。

```powershell
$P = "python"
$R = "D:\Projects\P1_DNA_loci_imaging"
$C = "deepblink_pytorch/scripts/diagnostics/scaffolds"
$O = "$R/docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103"
```

下表每项都有一行只读复跑命令。`runs/<UTC>/checks.json`记录13项实际命令与日志路径；最新成功记录由`HANDOFF.md`指向。创建示例输出时使用新的`--output`路径，写入函数拒绝覆盖。`bootstrap_scaffold.py`仅用于首次生成模板，不要在已有配置目录重复执行。

|项|新增/修改文件与产出|一行复跑命令|通过证据的含义|
|---|---|---|---|
|A1|`$C/train_mechanism.py`、`configs/S0.json`、`configs/P0.json`|`& $P "$C/train_mechanism.py" --config "$O/configs/P0.json" --dry-run`|schema_valid=true、launch_ready=false、adapter_implemented=false、optimizer_updates=0；S0也逐项验证|
|A2|`$C/capacity_control.py`、`configs/CAP_P0.json`和示例CAP_P0.json|`& $P "$C/capacity_control.py" --config "$O/configs/P0.json" --dry-run`|结构签名相同，激活/监督无新增参数，预期671719；不冒充新模型参数实测；未知容量变化会拒绝|
|A3|`$C/status.py`、`$C/04_status.sh`、`configs/registry.json`|`bash "$C/04_status.sh" --data-root "$R"`|S0/P0/CAP_P0 seeds42/43/44均not_started；完成训练仍41；不修改封存包旧04_status.sh|
|B4|`configs/ablation_matrix.json`|`& $P "$C/ablation_runner.py" --matrix "$O/configs/ablation_matrix.json" --dry-run`|合法组合校验，非法minus_signed被明确排除|
|B5|`$C/ablation_runner.py`|`& $P "$C/ablation_runner.py" --matrix "$O/configs/ablation_matrix.json" --dry-run`|打印6个去重后的dry-run命令即停，commands_executed=0|
|B6|`$C/aggregate_results.py`、示例per_seed_metrics.csv/model_summary.csv/schema.json|`& $P "$C/aggregate_results.py" --data-root "$R" --dry-run`|现有R0/H0/G0共18行primary/secondary导出、6组摘要；无新增训练结果；未开始方法的指标为空而非0|
|C7|`$C/formal_test_eval.py`、`configs/validation_model_locks.json`、`independent_test.json`|`& $P "$C/formal_test_eval.py" --data-root "$R" --locks "$O/configs/validation_model_locks.json" --config "$O/configs/independent_test.json" --dry-run`|9个验证选定权重/阈值锁核验；independent_test_images_read=0、test_evaluated=false；实时数据/推理适配器未启用|
|C8|`$C/ranked_ap.py`、`test_scaffolds.py`|`& $P -m pytest -q -s "$C/test_scaffolds.py" -k ranked_ap`|与现有scikit-learn1.8.0的AP积分核对；全部用合成数据，不生成baseline AP值|
|C9|`$C/weak_policy.py`、`configs/weak_policy.json`|`& $P "$C/weak_policy.py" --config "$O/configs/weak_policy.json" --features "$R/outputs/next_stage_v3_closing_controls/02_analysis/validation_truth_features.csv" --dry-run`|372个既有点，96个weak，旧标签逐项一致；无refit|
|D10|`$C/reproduce.py`和只读表复现示例|`& $P "$C/reproduce.py" --data-root "$R" --dry-run`|打印干净checkout+外部结果挂载的最短步骤；另外实际只读聚合跑通；未声称执行了新的git clone/checkout|
|D11|根README、实验登记、`$C/inspect_docs.py`|`& $P "$C/inspect_docs.py" --data-root "$R" --dry-run`|新增三项not_started且无metric_value；formal count=41|
|E12|`$C/blind_audit.py`、回填模板与列定义|`& $P "$C/blind_audit.py" validate --responses "$O/templates/blind_audit_response_template.csv" --key "$R/docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/overnight_ready/private/blind_audit_source_key.csv" --public-zip "$R/docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/overnight_ready/blind_audit_public.zip" --dry-run`|60ID无重复/缺漏，公开ZIP无私有表/列/额外成员/图像文本或EXIF；实际填写未完成|
|E13|`$C/blind_audit.py`聚合入口、假数据与agreement/crosstab示例|`& $P "$C/blind_audit.py" aggregate --synthetic-demo --dry-run`|两个SYNTHETIC审阅者、60个SIM匿名ID；示例绝不覆盖真实表，不报告真实人工一致率|

## A/B：容量和消融的解释

S0与P0暂定共享H0原结构：1通道输入、3通道头、无新增可训练参数，因此结构签名可证明参数预算一致。当前671719来自已存在H0的参数记录，不表示已实例化新的S0/P0网络。改变通道或增加参数的配置不会被“复制一个相同数字”伪装成容量匹配，生成器会明确失败。

P0−S0仅检验邻域一致监督；S0−H0隔离有符号表示范围。minus_support与capacity都是S0，不能当成两个独立实验重复训练。去掉signed范围却保留邻域监督可能产生不可表示目标，因此不是合法消融。完整的原论文base/capacity/full/minus四角色矩阵仍需审查；本骨架不虚构12次独立训练。

未来结果目录约定：`METHOD/seed_N/status.json`、`selection.json`、`primary/metrics.json`、`native_secondary/metrics.json`。结果CSV列由B6示例`schema.json`固定。缺失/未开始试验保留status和空指标；`ranked_ap`不允许用旧AP31冒充。

B6直接从既有metrics.json做聚合，而之前公开摘要从float32逐点重匹配后的距离归约。F1/P/R/弱点Recall均值与SD按1e-12核对；定位统计沿用先前分析的2e-7容差，实际最大差约1.28e-8px。逐字段差异保留在B6_gold_reproduction.json；没有改任何冻结数据或性能结论。

## C：正式评估定义

模型和阈值只从已冻结的validation selection读取，权重SHA、selection SHA和阈值绑定。当前曾被观察过的22图internal test不自动认定为独立test。新数据来源审查、manifest、完整候选缓存及实时推理适配器未接通，CLI因此只提供dry-run；候选缓存AP函数可独立测试，要求dataset-role、独立性确认、权重/数据manifest身份和全部prethreshold候选，无score floor。

完整ranked point AP：所有候选按全局score降序排序；同分按image ID/candidate index稳定排序；每个候选贪心占用同图半径3px内最近未匹配真值；同距离按真值索引打破。同分score group使用一个PR台阶，累加`ΔRecall × Precision`，不插值、不外推未达到的recall。无真值AP为未定义(None)，有真值无预测为0。该定义不是31点近似、COCO mAP或trapezoidal PR-AUC，也不替换既有F1的最大基数匹配。

AP积分层与[scikit-learn官方定义](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)及本机1.8.0实现比对；几何匹配另有手工可核算的合成案例，包括一个与Hungarian F1不同的反例。独立test正式执行前仍须把这项政策连同数据/模型锁预注册；本轮没有实际test AP。

弱点正式定义：原始未填充图像转换float32；在Python ties-to-even round后的真值中心取半径10的裁剪方窗，计算原像素中心值减窗均值、除以population SD(ddof0)。SD≤1e-6或特征非有限则unclassifiable，不归强点。`z < 2.4566830772038957`为weak，等于阈值为non_weak；匹配weak真值数/所有可分类weak真值数为弱点Recall。没有weak真值时Recall=None，另报unclassifiable数量。禁止用原policy_id悄悄改阈值，更改口径需新policy ID与明确审查。

## E：回填与一致率

盲审阶段填写候选是否真实spot、spot数量、置信度和审阅者；建议中心坐标若提供必须成对且在64×64原生patch范围内。实际检测TP/FP需要标注及一对一匹配上下文，只有解盲后才允许填写`adjudicated_detection_label`，并要求`adjudication_phase=post_blind_with_annotations`。不能把“附近有spot”直接等同于检测TP。

聚合分别输出每层spot-presence一致率和解盲后TP/FP一致率；分母只含双方可比较回答，缺失/uncertain显式列出，不能变成健康的零错误。另输出自动几何代理×人工判断的对照计数，不称代理为生物真值或据此计算“模型准确率”。同审阅者重复提交不作为两个独立审阅者。假数据使用SIM编号、SYNTHETIC_REVIEWER、synthetic_fixture=True，真实模式会拒收。

## 待TODO，不在本轮定参

- 待blind_audit回填/机制确认：P0邻域半径（候选2）、重叠归属规则、loss归一化规则、S0/P0是否进入训练、继续/停止门槛的正式锁定。
- 待实现并另立来源锁：有符号offset/邻域监督适配器、真实参数计数校验、合成forward/backward预检。**本轮不改网络实现或训练配方。**
- 待独立数据：新test manifest、采集/生物来源独立性审查、完整候选导出与实时推理适配器、正式评价执行授权。当前test_evaluated仍false。
- 待真实人工：60例逐项评分、第二审阅者与分歧裁定、解盲后TP/FP标签。所有示例统计都不代表人类结论。

## 一键核查全部骨架

```powershell
& $P "$C/run_scaffold_checks.py" --data-root "$R" --dry-run
```

每次另建`runs/<UTC>/`保存13份证据、示例和单测日志，最后复核362个受保护文件哈希。这个命令会真正执行只读聚合和合成测试；不会执行矩阵里打印的训练命令，不创建优化器，也不会读独立test图像。VS Code diff重点看注册表别名去重、非法消融拦截、AP匹配/同分规则、弱点边界值、盲审完成度与防泄漏检查。原模型性能结论保持不变。
