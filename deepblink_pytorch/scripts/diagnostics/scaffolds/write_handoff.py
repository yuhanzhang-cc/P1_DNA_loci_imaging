"""Publish an additive review index for one completed scaffold check run."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys
from scaffold_common import CODE_ROOT,base,read,table,csv_new,save_new


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--run",required=True)
    args=p.parse_args();root=args.data_root.resolve();out=base(root);run=out/"runs"/args.run;checks=read(run/"checks.json")
    assert checks["passed"] and len(checks["items"])==13
    old=table(root/"docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/model_summary.csv")
    new=table(run/"examples/aggregate_existing/model_summary.csv");count=0
    names=dict(R0="R0_cuda_reference",H0="H0_cuda_reference",G0="G0_ref_cuda");differences=[]
    for row in new:
        gold=next(r for r in old if r["method"]==names[row["method"]] and r["role"]==row["role"])
        for k,v in row.items():
            if k.endswith("_mean") or k.endswith("_sd"):
                tolerance=2e-7 if k.startswith("mean_error_px_") else 1e-12
                difference=abs(float(v)-float(gold[k]))
                assert difference<tolerance,(row["method"],k)
                differences.append(dict(method=row["method"],role=row["role"],metric=k,difference=difference,tolerance=tolerance))
                count+=1
    save_new(run/"evidence/B6_gold_reproduction.json",dict(passed=True,summary_rows=6,mean_sd_values_compared=count,
       core_metric_tolerance=1e-12,localization_tolerance=2e-7,
       sources="Direct source metrics.json reduction versus previously rematched float32 point-distance reduction; no frozen file changed",
       differences=differences,new_baseline_numbers=False))
    src=Path(__file__).resolve().parent
    paths={
      "A1":[src/"train_mechanism.py",out/"configs/S0.json",out/"configs/P0.json"],
      "A2":[src/"capacity_control.py",out/"configs/CAP_P0.json",run/"examples/CAP_P0.json"],
      "A3":[src/"status.py",src/"04_status.sh",out/"configs/registry.json"],
      "B4":[out/"configs/ablation_matrix.json"],"B5":[src/"ablation_runner.py"],
      "B6":[src/"aggregate_results.py",run/"examples/aggregate_existing/per_seed_metrics.csv",run/"examples/aggregate_existing/model_summary.csv",run/"examples/aggregate_existing/schema.json"],
      "C7":[src/"formal_test_eval.py",out/"configs/validation_model_locks.json",out/"configs/independent_test.json"],
      "C8":[src/"ranked_ap.py",src/"test_scaffolds.py"],"C9":[src/"weak_policy.py",out/"configs/weak_policy.json"],
      "D10":[src/"reproduce.py",src/"requirements-scaffold-qa.txt",src/"scaffold_common.py",src/"bootstrap_scaffold.py",src/"run_scaffold_checks.py",src/"write_handoff.py",run/"examples/clean_reproduction_table/model_summary.csv"],
      "D11":[CODE_ROOT/"README.md",CODE_ROOT/"docs/next_stage_plan_20261025/experiment_registry.csv",src/"inspect_docs.py"],
      "E12":[src/"blind_audit.py",out/"templates/blind_audit_response_template.csv",out/"templates/blind_audit_column_schema.json"],
      "E13":[run/"examples/synthetic_review"/n for n in ("synthetic_key.csv","synthetic_reviewer_1.csv","synthetic_reviewer_2.csv","agreement_by_stratum.csv","automatic_proxy_crosstab.csv","provenance.json")]}
    manifest=[]
    for item,files in paths.items():
        for f in files:
            relative=f.relative_to(CODE_ROOT) if f.is_relative_to(CODE_ROOT) else f.relative_to(root)
            manifest.append(dict(group=item[0],item=item,path=relative.as_posix(),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    csv_new(out/"FILES_BY_GROUP.csv",manifest,list(manifest[0]))
    snippets=dict(A1="schema_valid=true; launch_ready=false; adapter_implemented=false; optimizer_updates=0",
      A2="expected671719; measured_new_adapter=false; alias S0",A3="S0/P0/CAP_P0三种子not_started; 完成计数41",
      B4="非法minus_signed排除; base复用; minus_support别名去重",B5="6个命令仅打印; commands_executed=0",
      B6=f"18行现有结果/6组摘要; {count}个统计复现;核心容差1e-12/定位沿用2e-7",C7="9个validation锁核验; test图像读取0; test_evaluated=false",
      C8="sklearn1.8.0合成参考PASS;10 passed,12 deselected",C9="372点/96弱点;旧标签逐项相同; refitted=false",
      D10="打印干净checkout步骤;只读表复现通过; clean_checkout_executed=false",D11="三项not_started/指标空白;正式训练计数41",
      E12="60ID;映射无泄漏;responses_complete=false;240必填字段仍缺",E13="两份SYNTHETIC审阅者/60个SIM编号;真实盲审未完成")
    def link(f,label=None):return "["+(label or f.name)+"]("+os.path.relpath(f,out).replace("\\","/")+")"
    lines=["# A–E骨架验收与明早交接","","13/13 dry-run通过；22项单测通过；零GPU/优化器更新/新baseline/独立test图像读取。362个受保护文件哈希未变，正式训练计数仍41，既有结论不变。","",
       "S0/P0适配器和正式test实时数据/推理接口仍未实现。dry-run通过不表示可训练或已做独立测试。","",
       "完整清单："+link(out/"FILES_BY_GROUP.csv")+"；每项一行复跑命令及口径："+link(out/"README.md")+"。","",
       "|项|新增/修改文件与示例路径|证据摘要|日志|","|---|---|---|---|"]
    for item,files in paths.items():
        active=[root/f.relative_to(CODE_ROOT) if f.is_relative_to(CODE_ROOT) else f for f in files]
        lines.append("|"+item+"|"+"; ".join(link(f) for f in active)+"|"+snippets[item]+"|"+link(run/"evidence"/(item+".log"))+"|")
    lines += ["","完整命令/状态："+link(run/"checks.json")+"；单测："+link(run/"evidence/all_unit_tests.log")+"；B6逐值复现："+link(run/"evidence/B6_gold_reproduction.json")+"。","",
      "## 明早审查顺序","",
      "1. A3 "+link(out/"configs/registry.json")+"：CAP_P0=S0别名，不能重复计数；新04_status包装器不修改封存脚本。",
      "2. B4 "+link(out/"configs/ablation_matrix.json")+"：6次未来独立训练预算、合法去除和不可表示的非法组合。",
      "3. C9 "+link(out/"configs/weak_policy.json")+"：固定训练期阈值的严格小于边界、原图局部SD与unclassifiable处理。",
      "4. E12 "+link(out/"templates/blind_audit_response_template.csv")+"：先盲审填写；TP/FP裁定仅在解盲结合标注后填写。空白表不算审完。","",
      "## 留TODO的事项","",
      "- 待blind_audit/机制确认：P0邻域半径（候选2，正式值null）、归属规则、loss归一化、是否启动S0/P0、正式止损/继续规则。",
      "- 待另行实现：有符号offset/邻域监督适配器、真实参数计数和forward/backward预检；本轮不改网络或round2_protocol.py。",
      "- 待独立数据/协议：新test manifest、独立性和排重审查、完整候选导出、实时推理适配器与正式test执行。",
      "- 待真实人工：60例评分、第二审阅者、分歧裁定和解盲TP/FP；E13假数据不提供真实一致率结论。","",
      "## 一键复跑","",
      "python deepblink_pytorch/scripts/diagnostics/scaffolds/run_scaffold_checks.py --data-root D:\\Projects\\P1_DNA_loci_imaging --dry-run","",
      "每次另建UTC目录，不覆盖旧结果。私有结果需显式挂载；本轮未做新的git checkout或安装依赖，也未提交/推送，以便先审VS Code diff。QA requirements只定义可选隔离环境。"]
    with (out/"HANDOFF.md").open("xb") as stream:stream.write(("\n".join(lines)+"\n").encode("utf-8"))
    save_new(out/"environment.json",dict(python=sys.version,qa_runtime="existing base CPU",numpy="2.4.4",scipy="1.17.1",sklearn="1.8.0",pytest="9.1.1",installed_by_task=False))
    save_new(out/"pending_actions.json",dict(blind_audit_parameters=["P0.supervision.radius_px","P0.supervision.ownership","P0.supervision.normalization","review_decision"],
      mechanism_adapter_implemented=False,training_enabled=False,independent_test_data_ready=False,live_test_adapter_implemented=False,real_human_review_complete=False,completed_training_count=41))
    print(f"HANDOFF PASS: {len(manifest)} file entries; {count} reproduced statistics")


if __name__=="__main__":main()
