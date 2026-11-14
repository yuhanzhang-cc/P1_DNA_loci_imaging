# 机制诊断与过夜准备汇报

第1步结果位于上级目录的`peak_offset_diag.csv`与`CONCLUSION.md`；它们保留该步骤的输入、符号、边界/SNR分层、来源峰/offset分解和复跑命令。此文件反映用户要求继续后的第2–4步准备状态。

## 目前结论

- 保留G0。H0的负偏移主要出现在来源峰不等于监督锚点的663个TP事件；正确锚点251个事件几乎无平均偏移。序列化最大误差1.53e-5px、CPU/GPU最大差3.05e-5px，远小于约−0.23px的偏移。
- 627/914个H0 TP在固定来源峰的`[0,1]²`输出范围下无法精确到达真值。这是可核算的表示范围限制，支持峰位置—offset监督配对不足的机制假说。它不证明所有误差均由单一bug导致，也不能被当作应该做常数坐标平移的证据。
- 65和59必须解释成**净差**。逐个空间对应后为共有83、H0独有147、G0独有82。147个H0独有FP中78为单真值近邻重复代理、69待视觉判断；该53.1%也不能被称为已确认的生物误检原因。此前90.8%只是类别净增/总数净增之比，不是独有候选的占比。
- 未发现多真值分配冲突代理案例；原先合并名称中的“冲突”不能被写成已观测原因。原始近邻候选可能是重复检出，也可能涉及未标注真spot，需盲审判定。
- 已生成60例匿名审阅包；**尚无人类评分**。不能给“真实背景纹理”等标签制造概率置信度。当前置信度字段区分确定的几何规则与未评估的生物学判断。

逐条规则：距一个真值≤3px且该真值已被其他候选匹配，标为近邻重复代理；距多个真值≤3px，标为分配冲突代理；无3px近邻的候选只标为待视觉复核，另记3–6px或>6px距离层。`geometry_rule_confidence`表示确定执行了规则，不是生物类别的概率。真实背景纹理需人工确认，`biological_confidence`目前均为unassessed。

公开图只给出原始视图和单个待审候选圈，供人判断是否真实spot、spot数量和位置；审阅者不应仅凭单圈图片猜测“模型重复次数”。锁定回答后，负责人再结合几何配对与私有来源键裁定重复、标注不足或背景等解释。

## 文件、输入口径和复跑

|产出|口径/输入|路径|
|---|---|---|
|FP逐条表|H0全部230个已验收FP，标明是否属于147个空间独有事件、83个共有事件；同时保留原几何标签|`h0_extra_fp_breakdown.csv`|
|FP对账|同种子同图像，3px最大基数一对一FP匹配；1/2/3px敏感性不重选阈值|`fp_reconciliation.json`、`fp_shared_pairs.csv`、`g0_only_fp.csv`、`fp_spatial_sensitivity.csv`|
|可表示范围|由上级逐TP表的真值和CPU来源锚点计算到闭区间盒`source+[0,1]²`的距离下界|`peak_offset_source_range.csv`、`peak_offset_source_range_summary.json`|
|公开审阅表|随机种子20261102；三层各20例；来源、模型与层别均隐藏|`blind_audit_public/blind_audit_manifest.csv`|
|图像包|64×64原生裁剪；固定整图zscore窗口[-1,8]；raw/marked双视图；无物理标尺|`blind_audit_public.zip`、`.sha256`；公开ZIP123文件|
|私有映射|匿名ID到模型、种子、层别、图像、预测索引、crop坐标；不得给盲审者|`private/blind_audit_source_key.csv`|
|定量裁剪与显示元数据|原始像素float32裁剪，不重采样；窗外缺失像素NaN|`private/quantitative_patches/`、`private/source_image_manifest.csv`|
|QA|60个定量裁剪、60个显示变换和60个候选标记逐像素核验；输入哈希不变，精确显示图重复0|`qa_checks.json`|
|任务设计|自动重放、人工依赖、未启用的S0/P0实验提案|`SPEC_OVERNIGHT_20261103.md`|

重建第2–3步及范围检查（已存在目录不覆盖，指定新子目录）：

```powershell
python deepblink_pytorch/scripts/diagnostics/prepare_overnight_review.py --data-root D:\Projects\P1_DNA_loci_imaging --output D:\Projects\P1_DNA_loci_imaging\docs\next_stage_plan_20261025\baseline_cuda_analysis_20261102\diagnostics\overnight_ready_replay
```

复跑当前包QA：

```powershell
python deepblink_pytorch/scripts/diagnostics/validate_overnight_review.py --data-root D:\Projects\P1_DNA_loci_imaging
```

检查过夜就绪状态：

```powershell
python deepblink_pytorch/scripts/diagnostics/run_overnight_no_train.py --data-root D:\Projects\P1_DNA_loci_imaging --validate-ready
```

整条无训练运行命令见SPEC。所有命令均用现有`spotiflow_cpu`环境，不修改共享依赖。除已声明的窗口映射/裁剪/候选圈外没有图像增强；人工审阅前应保持私有映射关闭。完整SIFT旋转/裁剪复用检查因缺OpenCV未运行，不能将精确重复为0宣称成所有复用情况均不存在。

## 待决策

暂定后续仅检验“有符号offset范围 + 峰邻域一致监督”，用等参数S0对照归因；方案未实现或训练。人工盲审完成后再决定是否进入该分支。当前不会把H0升级为核心底座，也不会把范围/监督修复预先称为论文创新。
