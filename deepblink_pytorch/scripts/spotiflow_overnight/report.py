"""Evidence-separated morning report; partial training is never a three-seed result."""
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ROOT, OUT, REPORT, rows, dump, now


def write_report(stage="engineering_complete"):
    summaries=[]
    for key in ("R0","G0"):
        source=rows(ROOT/"outputs/next_stage_v3_closing_controls/validation_comparison.csv")
        selected=[r for r in source if r["model"]==key]
        vals=[float(r["f1"]) for r in selected]
        summaries.append(dict(method=key,track="local_training",n=3,values=vals,mean=statistics.mean(vals),sd=statistics.stdev(vals)))
    for name,track in (("Z0_default","Z0 native defaults"),("Z1_validation_calibrated","Z1 calibrated transfer")):
        path=OUT/"01_pretrained"/name/"selection.json"
        if path.exists():
            result=json.loads(path.read_text())
            value=result["metrics"]["f1"]
            summaries.append(dict(method="Spotiflow "+name[:2],track=track,n=1,values=[value],mean=value,sd=None))
    complete=[]
    for seed in (42,43,44):
        path=OUT/f"02_finetune/seed_{seed}/status.json"
        if path.exists() and json.loads(path.read_text())["status"]=="complete":
            selection=json.loads(path.with_name("selection.json").read_text())
            complete.append(dict(seed=seed,**selection["primary"]))
    if len(complete)==3:
        vals=[r["f1"] for r in complete]
        summaries.append(dict(method="Spotiflow F",track="native_finetuning",n=3,values=vals,mean=statistics.mean(vals),sd=statistics.stdev(vals)))
    destination=OUT/"06_report"
    destination.mkdir(parents=True,exist_ok=True)
    dump(destination/"summary.json",dict(methods=summaries,completed_finetune_seeds=complete,test_evaluated=False))
    with plt.rc_context({"font.family":"DejaVu Sans","svg.fonttype":"none"}):
        fig,ax=plt.subplots(figsize=(10,6))
        fig.subplots_adjust(bottom=.24,left=.1,right=.96,top=.86)
        colors=["#666666","#007F60","#75A5BE","#0072B2","#A66B00"]
        for i,row in enumerate(summaries):
            ax.bar(i,row["mean"],color=colors[i],edgecolor="#333333",width=.6,
                   yerr=row["sd"] if row["n"]==3 else None,capsize=4,hatch="//" if row["n"]==1 else None)
            for j,value in enumerate(row["values"]):
                ax.plot(i+(j-(row["n"]-1)/2)*.12,value,"o",color="#222222",markerfacecolor="white",markersize=4)
            ax.text(i,row["mean"]+(row["sd"] or 0)+.03,f"{row['mean']:.4f}",ha="center",fontsize=10)
        ax.set(xticks=range(len(summaries)),xticklabels=[r["method"] for r in summaries],ylim=(0,1.),ylabel="Development micro-F1")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        fig.suptitle("Development references: local training and pretrained transfer",fontsize=14,weight="bold")
        fig.text(.1,.13,"R0/G0: mean ± SD of 3 training seeds. Hatched bars: one frozen pretrained checkpoint, no seed SD.",fontsize=9)
        fig.text(.1,.08,"Z0: native defaults; Z1: validation-calibrated threshold. Neither is a fully adapted competitor.",fontsize=9)
        fig.text(.1,.03,"Same 22 validation images; reused for model selection. No independent test or significance claim.",fontsize=9)
        for ext in ("png","svg"):fig.savefig(destination/f"development_reference_comparison.{ext}",dpi=300,facecolor="white")
        plt.close(fig)
    pilot_path=OUT/"00_engineering/fullsize_cpu_pilot/runtime.json"
    pilot=json.loads(pilot_path.read_text()) if pilot_path.exists() else {}
    testdirs=sorted((OUT/"00_engineering").glob("recovery_check_*/checks.json"))
    tested=json.loads(testdirs[-1].read_text()) if testdirs else {}
    review_path=OUT/"03_error_review/checks.json"
    review=json.loads(review_path.read_text()) if review_path.exists() else {}
    lines=["# 2026-10-30 过夜工程与开发比较工作汇报","",f"更新时间：{now()}（America/New_York）。阶段：{stage}。","",
        "## 已完成的工程与证据","",
        f"- 恢复预检：{'通过' if tested.get('passed') else '未完成'}；连续与恢复模型最大差：{tested.get('model_max_abs','待验证')}；优化器、调度器和原生权重重载独立检查。",
        "- 独立实现原生Z0与冻结权重Z1；相同22张完整原图和公共3px最大匹配，不用裁块F1替代主评价。",
        "- 正式微调工程使用原生数据/损失/增强；固定256 crop、batch4、AdamW lr0.0003、wd0.01、无放回逐图采样，44更新/轮。",
        "- 每5轮保存真实当轮模型，20×31校准；保存完整Lightning恢复状态、Python/NumPy/Torch/设备/加载器RNG。增强按seed/epoch/image确定，支持epoch边界精确恢复。",
        "- 未变更共享XPU环境、旧训练器、原始数据、标签、原划分；没有读取测试数组。","",
        "## 当前开发结果","","| 方法 | 轨道 | 重复定义 | F1 |","|---|---|---|---:|"]
    for row in summaries:
        value=f"{row['mean']:.4f}"+(f" ± {row['sd']:.4f}" if row["sd"] is not None else "")
        lines.append(f"| {row['method']} | {row['track']} | {'3训练种子' if row['n']==3 else '1预训练checkpoint'} | {value} |")
    lines += ["","Z0/Z1是公开权重迁移参照；G0的较高开发分数不能替代Spotiflow本地适配后的公平比较。三种子单配置也不等于充分调优或全面SOTA。","",
              "## CPU成本与正式训练状态", "",
              f"实测更新中位耗时：{pilot.get('median_update_seconds_after_two_warmups',0):.3f}秒；三种子预计{pilot.get('estimated_three_seed_hours',0):.2f}小时，加25%余量后{pilot.get('estimated_three_seed_hours_with_25pct_margin',0):.2f}小时。",
              f"预定CPU训练窗口关口为5.5小时；是否满足：{pilot.get('run_three_seeds_on_cpu',False)}。已完成正式微调种子数：{len(complete)}/3。",
              "CPU过慢不代表方法低准确率。正式微调未完成时，不输出三种子平均或关于适配后优劣的最终结论。","",
              "## 自动审查与人工工作", "",
              f"原40图复核队列已生成材料：{review.get('queue_images',0)}张；自动错误候选：{review.get('error_candidate_rows',0)}条。人工审核：未完成。",
              "匿名方法叠图/patch及复核索引见03_error_review；所有FP/FN相对于现有TrackMate参考，疑似漏标不自动确认为真实位点。",
              "215源容器中有两个跨train/valid，真实生物学独立性与预训练同源风险尚未解决；不自动改划分。",
              "同CPU、同线程和精度的端到端时间见04_efficiency。该表不以CPU外部方法与XPU本地方法混排速度。","",
              "## 修复与剩余事项", "",
              "1. 合成极小网络的单卷积配置触发原生slim-UNet通道错误；预检采用受支持的三卷积块，general架构不变；失败证据保留。",
              "2. 固定样本/轮次增强及独立sampler，补足原生fit缺少ckpt_path入口及仅保存native-best的问题；未修改第三方安装文件。",
              "3. 正式三种子微调仍需按设备决定推进；CUDA材料仅在本地准备，没有租赁、上传或云端运行。",
              "4. 独立标注复核、新数据验证、平衡调优和核心架构创新关口仍未完成；本次不追加注意力模块或搜索直到获胜。","",
              "可复现入口：deepblink_pytorch/scripts/run_spotiflow_overnight.py；状态和事件日志见当前输出目录。",
              "图件保持均值±SD和实际重复；不为得到星号改检验单位。",
              "绘图/统计流程参考：Kassis T., Agarwal V., He Y., Patel D., Brueckner A. M. (2026), Scientific Agent Skills, https://doi.org/10.48550/arXiv.2609.00065 。"]
    text="\n".join(lines)+"\n"
    (OUT/"morning_report.md").write_text(text,encoding="utf-8")
    REPORT.write_text(text,encoding="utf-8")
    return summaries
