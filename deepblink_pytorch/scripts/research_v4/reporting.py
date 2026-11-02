"""Evidence-derived Night-L report; pending GPU/human gates stay explicit."""
from pathlib import Path
from .common import read_json,rows,dump,sha256,verify_protected,source_hashes,user_time


def report(ctx):
    out=ctx.out;docs=ctx.code_root/"docs/next_stage_plan_20261025"
    audit=read_json(out/"00_protocol/input_audit.json")
    geometry=read_json(out/"01_preflight/augmentation_checks.json")
    cpu=read_json(out/"01_preflight/cpu_engineering_v2/checks.json")
    native=read_json(out/"01_preflight/spotiflow_scratch_engineering/checks.json")
    classical=read_json(out/"02_classical/checks.json")
    scaffold=read_json(out/"06_study_scaffold/checks.json")
    review=read_json(ctx.data_root/ctx.spec["review"]["output"]/"checks.json")
    bundles=read_json(out/"04_bundles/latest.json")
    verification_path=out/"05_verification/checks.json"
    verification=read_json(verification_path) if verification_path.exists() else dict(passed=False,reason="standalone bundle verification pending")
    for check in (geometry,cpu,native,classical,scaffold,review,bundles):
        if not check["passed"]:raise ValueError("Do not write a completed report from failed checks")
    verify_protected(ctx.data_root,read_json(out/"00_protocol/protected_hashes.json"))
    raw=read_json(out/"00_protocol/raw_development_hashes.json")
    if any(sha256(ctx.data_root/n)!=h for n,h in raw.items()):raise ValueError("Raw development files changed")
    result=dict(status="local_preparation_complete" if verification["passed"] else "partial_with_failures_or_deferred_tasks",
        updated_user_time=user_time(ctx.data_root),formal_training_tasks_started=0,new_data_collected=False,test_arrays_loaded=False,
        raw_files_unchanged=len(raw),protected_artifacts_unchanged=audit["protected_artifacts"],human_review_completed=False,
        synthetic_optimizer_updates_total=25,discarded_real_optimizer_updates=2,
        default_gpu_tasks_deferred=6,optional_baseline_gpu_tasks_deferred=9,core_ablation_enabled=0,
        bundle_verification=verification,source_hashes=source_hashes(ctx.code_root))
    dump(out/"07_report/completion_checks.json",result)
    dump(out/"00_protocol/source_hashes_final.json",source_hashes(ctx.code_root))
    table=[]
    for r in rows(out/"02_classical/summary.csv"):
        table.append(f"| {r['method']} | {float(r['f1']):.4f} | {float(r['precision']):.4f} | {float(r['recall']):.4f} | {float(r['weak_recall']):.4f} | {r['configuration_threshold_pairs']} |")
    package_lines="\n".join(f"- `{Path(b['path']).name}`：{b['bytes']/1024**2:.1f} MiB；SHA256 `{b['sha256']}`。" for b in bundles["archives"])
    text=f"""# 2026-11-02 过夜任务完成报告

更新时间：{result['updated_user_time']}。本轮状态：**{result['status']}**。

## 交付与范围

用户在原SPEC后授权执行并继续搭建全课题流程；需要GPU训练或新数据采集的任务整理后暂缓。本轮新增正式训练为**0**，新采集数据为**0**，没有重新读取22张内部测试图。原SPEC和参数草案保留其提案状态；实际完成情况以本报告、执行日志和manifest为准。

已完成增强/恢复运行器、LoG和DoG完整扫描、60图新版审查包、12个现有模型适配检查、29项历史训练索引、GPU分队列、固定阈值外部评价与统计脚本、两个私有上传包。源码在`codex/overnight-v4-preparation`安全工作树开发，没有改写历史训练核心文件。

## 本轮可报告的结果

同22张开发验证图、3px匹配，两个传统参考各24配置×31阈值，完整评价1488组：

| 方法 | micro-F1 | Precision | Recall | 弱点Recall | 配置阈值组合 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

LoG/DoG是本项目固定尺度空间实现，不称为他人官方默认结果。阈值候选由训练图响应分位数生成，再在验证集选配置；无伪三种子或虚构SD。弱点参考固定96/372点。

历史三种子开发F1：R0 **0.8065±0.0021**、G0 **0.8251±0.0038**、Spotiflow适配 **0.8606±0.0034**，均为sample SD。这些成绩没有因本轮工程变更而更新。当前没有新增的自有模型性能提升证据；传统对照较弱也不能代替与适配Spotiflow的强比较。开发集用于模型与阈值选择，统计区间未校正该选择，不能作为独立泛化证明。

## 验收证据

- 198张开发图/4528点，396个原始TIF/CSV文件和62个历史保护文件最终复核未变；仅8个开发NPZ键。两处训练/验证来源容器重叠仍需核实生物学关系；198图未发现精确像素重复，未完成近重复/裁剪重复鉴定。
- 1584个D4案例通过，最大坐标往返误差`{geometry['inverse_max_error_px']}`px；12次浮点边界修正逐项留痕，原始标注未改。强度扰动在有效区域zscore后施加，padding保持0，模型/采样RNG不被增强消耗。
- 初版16次合成更新、最终版8次合成更新、Spotiflow scratch1次，共25次合成工程更新，低于32上限；另外仅各1个256/512真实批次，均丢弃。12模型仅前后向检查，没有优化器更新。
- 最终恢复检查模型最大差为0、优化器/RNG一致，阈值与权重绑定；模拟“快照写完但恢复点未提交”时保留孤立快照后精确重放。正式100轮/2200或4400更新尚未在本机执行。
- 新审查包为38训练+22验证=60图、180方法面板、60张patch表、810条参考相对错误候选。固定seed42，Spotiflow参照为真实CUDA适配主模型；统一每图显示窗、原生像素，无猜测微米比例尺。人工审查180项仍pending，不能把候选当作标注错误结论。
- 独立数据入口检查SHA、像素单位、尺寸、标签表头、来源和生物学组，拒绝开发图精确重复；固定模型/阈值后评价，一次读出保留。元数据合格不自动证明生物学独立。
- 独立解压包、所有manifest文件、Bash语法、模型重载、评价导出和统计检查：**{'通过' if verification['passed'] else '尚未全部通过'}**，详见`05_verification/checks.json`。CUDA、Linux依赖安装和租用设备实测留给服务器，未声称已经通过。

## 异常及修复

1. Windows短暂文件锁导致经典对照CSV原子替换失败：保留失败日志，增加有限重试，用同一算法锁/候选缓存恢复，最终1488组完整。旧`failure.txt`为已解决事件，当前状态以completion_checks为准。
2. Spotiflow配置直接从YAML构造触发list/tuple校验失败：改用官方配置加载入口，随机初始化检查通过，没有加载预训练权重。
3. 长墙钟间隔与CPU累计时间显著不符，原因未确定，可能涉及宿主休眠或调度。增加临时系统唤醒请求和CPU/墙钟双日志；结束时释放。经典缓存恢复的elapsed不是完整推理耗时，不用于效率排名。
4. 增加按源代码/manifest绑定的检查、各队列实际CUDA输入检查、未提交快照隔离与恢复，避免用CPU成功替代GPU验收。

## 下一步顺序

1. **先跑recipe六次**：G0参考/联合D4+强度增强×seed42/43/44，同CUDA、100轮、batch8、float32、无AMP/TF32、每任务620次选择。只检验联合训练配方是否有用，不能拆称几何或强度各自贡献，也不是架构创新。
2. 按需跑九次baseline：R0三次、H0三次、Spotiflow scratch三次，独立启动。scratch是相同架构/原适配配方下的预训练控制，不能称已充分调优的scratch最佳模型。R0/H0/G0与Spotiflow原生损失、裁剪、优化不同，属于方法级比较，不是纯架构因果消融。
3. 完成人工审查，优先分清弱信号漏检、密集点网格冲突、背景误检及标注争议；若配方确有收益，再拆几何/强度消融。没有证据时不继续堆注意力。
4. 核心方法仍未选定。四角色×三种子的base/capacity/full/minus矩阵已建并校验，但**0项启用**；需先锁瓶颈、相关工作差异、容量对照和可训练的机制移除，才能登记后续实验。不能把空模板或旧R1结果当成新贡献已完成。
5. 用户采集有真实生物学/采集批次身份的新图，完成标注规范与盲审后，使用外部包九份固定模型一次评价。数据质量、独立性与样本规模需要实际设计，模板不是数据。

三训练种子的双侧精确符号翻转p值最低0.25；不为获得星号反复训练或把图像当独立生物学重复。汇总脚本输出均值±SD柱图、配对线图、预声明对比及Holm校正，外部数据按生物学组重采样；不能保证显著或预定“我们最好”。

## 私有上传包

包目录：`outputs/next_stage_v4_recipe_20261101/04_bundles/{bundles['source_revision']}/`。

{package_lines}

完整命令、回传及优先级见[GPU交接指南](gpu_handoff_20261102.md)。审查入口为本地`outputs/next_stage_spotiflow_20261030/08_finetuned_error_review/index.html`。原始图、NPZ、检查点、身份映射和逐图预测均留本地，不递归发布outputs。
"""
    docs.mkdir(parents=True,exist_ok=True)
    (docs/"overnight_report_20261102.md").write_text(text,encoding="utf-8")
    (docs/"overnight_progress_20261102.md").write_text("# 本轮进度\n\n"+f"更新时间：{result['updated_user_time']}；状态：{result['status']}。\n\n"+
        "本地工程已完成，GPU训练与新数据采集等待用户执行；人工复核pending，核心方法未选定。\n\n"+
        "[完整汇报](overnight_report_20261102.md) · [GPU命令](gpu_handoff_20261102.md)\n\n"+
        "机器进度：`outputs/next_stage_v4_recipe_20261101/controller_status.json`；最终验收：`07_report/completion_checks.json`；逐阶段事件：`execution_events.jsonl`。\n",encoding="utf-8")
    ctx.task("final_report","completed",status_report=result["status"])
    return result
