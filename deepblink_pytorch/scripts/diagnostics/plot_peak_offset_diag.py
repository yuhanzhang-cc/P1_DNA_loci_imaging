"""Plots and reproducible step-1 report; no inference or model modification here."""
from pathlib import Path
import hashlib
import json
import csv
import numpy as np


def load_diag(out):
    with (out / "peak_offset_diag.csv").open(encoding="utf-8", newline="") as stream:
        data = list(csv.DictReader(stream))
    boolean = ("common_tp", "cpu_source_linked", "peak_equals_target_anchor", "peak_anchor_occupied_by_other_truth", "offset_at_target_anchor_only_gt_conditioned")
    text = ("method", "sample_id", "region")
    for row in data:
        for key, value in list(row.items()):
            if key in text:
                continue
            if key in boolean:
                row[key] = value == "True" if value else None
            else:
                row[key] = float(value) if value else None
    return data, json.loads((out / "provenance.json").read_text(encoding="utf-8"))


def render(out, data, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(H0="#0072B2", R0="#D55E00")
    with plt.rc_context({"svg.fonttype": "none", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), layout="constrained")
        for column, axis in enumerate(("y", "x")):
            ax = axes[0, column]
            for method, style in (("H0", "-"), ("R0", "--")):
                values = np.asarray([r[f"d{axis}_px"] for r in data if r["method"] == method])
                ax.hist(values, bins=metadata["hist_edges"], density=True, histtype="step", lw=1.7,
                        linestyle=style, color=colors[method], label=f"{method}: n={len(values)}, mean={values.mean():+.3f}")
                ax.axvline(values.mean(), color=colors[method], linestyle=style, lw=1)
            ax.axvline(0, color="0.5", lw=.7)
            ax.set(xlim=(-3, 3), xlabel=f"Exported prediction - truth, d{axis} (px)", ylabel="Density (1/px)", title=f"{'AB'[column]}  All accepted TPs, {axis} axis")
            ax.legend(frameon=False, fontsize=9)
            ax = axes[1, column]
            linked = [r for r in data if r["method"] == "H0" and r["cpu_source_linked"]]
            shifts = [r[f"peak_anchor_shift_{axis}_px"] for r in linked]
            residuals = [r[f"d{axis}_px"] for r in linked]
            ax.scatter(shifts, residuals, s=12, alpha=.22, color=colors["H0"], edgecolor="none", rasterized=True)
            ax.axhline(0, color="0.5", lw=.7); ax.axvline(0, color="0.5", lw=.7)
            ax.set(xlim=(-4, 4), ylim=(-3, 3), xticks=range(-4, 5), ylabel=f"Exported d{axis} (px)",
                   xlabel=f"H0 source peak anchor - GT floor anchor, {axis} (px)", title=f"{'CD'[column]}  Source-attributed H0 TPs, n={len(linked)}")
        fig.suptitle("Frozen primary models: exported bias and directly observed source anchors", fontsize=13)
        for extension in ("png", "svg"):
            fig.savefig(out / ("peak_offset_overview." + extension), dpi=300)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
        for column, axis in enumerate(("y", "x")):
            ax = axes[column]
            for method, marker, style in (("H0", "o", "-"), ("R0", "s", "--")):
                means = []
                for seed in (42, 43, 44):
                    values = [np.mean([r[f"d{axis}_px"] for r in data if r["method"] == method and r["seed"] == seed and r["common_tp"] and r["snr_quartile"] == q]) for q in (1, 2, 3, 4)]
                    means.append(values)
                    ax.plot(range(1, 5), values, color=colors[method], linestyle=style, marker=marker, alpha=.25, lw=.8, ms=3)
                means = np.asarray(means)
                ax.errorbar(range(1, 5), means.mean(0), yerr=means.std(0, ddof=1), color=colors[method],
                            linestyle=style, marker=marker, lw=1.7, capsize=3, label=method)
            ax.axhline(0, color="0.5", lw=.7)
            ax.set(xticks=(1, 2, 3, 4), xticklabels=("Q1 low", "Q2", "Q3", "Q4 high"),
                   xlabel="Fixed truth-local-zscore quartile", ylabel=f"Mean common-TP d{axis} (px)", title=f"{'AB'[column]}  Mean across seeds +/- sample SD")
            ax.legend(frameon=False)
        for extension in ("png", "svg"):
            fig.savefig(out / ("peak_offset_snr." + extension), dpi=300)
        plt.close(fig)
    report(out, data, metadata)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != "artifact_hashes.json"}
    (out / "artifact_hashes.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))


def report(out, data, meta):
    def selected(method="H0", common=False, region=None, q=None):
        return [r for r in data if r["method"] == method and (not common or r["common_tp"])
                and (region is None or r["region"] == region) and (q is None or r["snr_quartile"] == q)]
    def mean(group, key):
        values = [r[key] for r in group if r.get(key) is not None]
        return f"{np.mean(values):+.6f}" if values else "NA"
    def quant(group, key):
        values = [r[key] for r in group if r.get(key) is not None]
        return "/".join(f"{v:+.3f}" for v in np.quantile(values, [.05,.5,.95])) if values else "NA"
    h0 = selected()
    linked = [r for r in h0 if r["cpu_source_linked"]]
    command = r"python deepblink_pytorch/scripts/diagnostics/peak_offset_diag.py --data-root D:\Projects\P1_DNA_loci_imaging"
    replot = command + " --plots-only"
    lines = ["# 第1步：冻结模型峰值/offset 对齐诊断", "",
             "仅执行本步骤。未启动训练、修改架构/round2_protocol.py、重选阈值或校正坐标。第2–4步等待用户确认。", "",
             "## 输入与计算口径", "",
             "- 原始验收输入：extracted_h0/extracted_r0 中三个种子的 primary/prediction_points.csv、per_image_metrics.csv、matches.csv。",
             "- 真值及图像：原冻结训练包 assets/development_only.npz，仅读取四个 *_valid 数组；没有读取训练或测试数组。",
             "- 原始峰/offset 补充输入：上述六个 primary 选中权重 best_model.pt，只读 CPU float32 推理，所有参数及输入文件前后哈希相同。",
             "- 最终偏移始终用已验收 GPU 导出坐标减真值，负dy向上、负dx向左。共同TP按同种子同图同真值索引配对。",
             f"- CPU来源映射：按原候选顺序核对，必要时唯一坐标匹配，半径{meta['cpu_source_matching_radius_px']}px；缺失映射保留，不能从最终坐标floor反推来源。",
             f"- SNR采用既有真值点 local_zscore，即半径10（至多21×21）的局部像素强度z分数，不能当作物理SNR；372个唯一真值点的共同四分位切点为 {meta['snr_quartile_cuts']}。",
             "- 边界：真值到原始图像FOV边缘距离<10px；中心为其余。3px/32px只作范围敏感性检查，不用填充后画布界定区域。",
             "- 逐点事件跨种子重复，不能当作独立生物样本。均值/分位数只作诊断描述，不新增显著性检验。", "",
             "## 结果", "", "|模型/群体|TP事件数|唯一真值点|平均dy，px|平均dx，px|", "|---|---:|---:|---:|---:|"]
    for method in ("H0", "R0"):
        for common in (False, True):
            group = selected(method, common)
            lines.append(f"|{method}/{'共同TP' if common else '全部TP'}|{len(group)}|{len({(r['sample_id'],r['true_index']) for r in group})}|{mean(group,'dy_px')}|{mean(group,'dx_px')}|")
    lines += ["", "### 区域与信号分层：共同TP", "", "|层|事件数|唯一真值点|H0 dy / dx，px|R0 dy / dx，px|", "|---|---:|---:|---|---|"]
    for label, region, q in [("边界<10px", "boundary", None),("中心", "center", None)] + [("信号Q"+str(q),None,q) for q in (1,2,3,4)]:
        a,b=selected(common=True,region=region,q=q),selected("R0",common=True,region=region,q=q)
        lines.append(f"|{label}|{len(a)}|{len({(r['sample_id'],r['true_index']) for r in a})}|{mean(a,'dy_px')} / {mean(a,'dx_px')}|{mean(b,'dy_px')} / {mean(b,'dx_px')}|")
    for band in (3,10,32):
        a=[r for r in h0 if r["edge_distance_px"]<band]
        lines.append(f"\n边界<{band}px：H0共{len(a)}个TP事件，{len({(r['sample_id'],r['true_index']) for r in a})}个唯一真值点；dy={mean(a,'dy_px')}，dx={mean(a,'dx_px')}px。样本稀少时不解释为区域效应。")
    lines += ["", "### H0来源峰与监督锚点", "",
              f"H0 {len(h0)}个TP中有{len(linked)}个找到CPU原始候选来源；其中{sum(r['peak_equals_target_anchor'] for r in linked)}个来源峰恰落在本真值floor锚点。",
              "来源峰指热图局部极大值的整数像素，而不是原图强度峰。offset监督只在真值floor锚点上；邻近锚点取到的offset不是本真值的监督位置。", "",
              "|轴|平均峰锚点位移|峰位移 P05/中位/P95|平均offset误差|offset误差 P05/中位/P95|平均float32误差|最大绝对float32误差|", "|---|---:|---|---:|---|---:|---:|"]
    for axis in ("y","x"):
        cast=[abs(r[f"float32_cast_error_{axis}_px"]) for r in linked]
        lines.append(f"|{axis}|{mean(linked,f'peak_anchor_shift_{axis}_px')}|{quant(linked,f'peak_anchor_shift_{axis}_px')}|{mean(linked,f'offset_error_{axis}_px')}|{quant(linked,f'offset_error_{axis}_px')}|{mean(linked,f'float32_cast_error_{axis}_px')}|{max(cast,default=0):.9g}|")
    for kind, group in (("源峰等于目标锚点", [r for r in linked if r["peak_equals_target_anchor"]]),("源峰偏离目标锚点",[r for r in linked if not r["peak_equals_target_anchor"]])):
        lines.append(f"\n{kind}：{len(group)}事件；dy={mean(group,'dy_px')}，dx={mean(group,'dx_px')}px。")
    lines += [f"\n在真值floor锚点读取offset时，平均有符号误差为dy={mean(linked,'offset_error_at_gt_anchor_y_px')}、dx={mean(linked,'offset_error_at_gt_anchor_x_px')}px。该读取使用真值位置，仅用于诊断，不能视为部署性能。",
              f"\n明确观察到：{len(linked)-sum(r['peak_equals_target_anchor'] for r in linked)}/{len(linked)}个H0 TP的来源峰不在目标监督锚点。来源峰正确的组几乎无平均偏移，来源峰错位组的双轴平均偏移约−0.31px。中心区域和四个信号分位均保留负偏移；10px边界组共同TP只有1个，不能据其判断区域效应。"]
    lines += ["", "逐点恒等分解：最终GPU残差 = 来源峰锚点位移 + 来源峰offset相对目标offset的误差 + float32序列化误差 + GPU导出与CPU重放差异。每个可链接事件均做数值恒等校验。",
              "目标锚点处另读的offset仅作使用真值位置的诊断字段；没有把它替换进预测，没有计算其F1，也不能把这种真值条件结果当作可部署算法。", "",
              "### 与边界舍入修复的分离", "",
              "- rounding_audit.csv统计六个最终选中模型CPU重放时、阈值过滤前所有候选的float64→float32上界round-out；不是把训练期间某轮的事件冒充最终模型事件。",
              "- 已归档训练恢复事件仅为seed43/epoch60的一个超低分边界候选；其checkpoint不是最终选中的epoch85。",
              "- 表中float32误差直接从原始锚点+offset与存储坐标相减得到；中心区域残差和大于该误差尺度的峰/offset项可与FOV边界过滤分开观察。",
              "- sigmoid offset饱和为0/1与最终坐标序列化的round-out是不同现象，不能混为一个浮点bug。", "",
              "## 结论与证据边界", "",
              f"冻结导出复核H0平均dy={mean(h0,'dy_px')}、dx={mean(h0,'dx_px')}px。已直接记录预测来源峰与监督锚点的位移，并将其与offset误差、CPU/GPU差异和序列化误差逐项分离。",
              "这些观测可以检验“峰位置与单锚点offset监督不一致”的假说，但单靠来源分解不能证明该训练设计就是全部偏移的因果来源，也不能据此进行验证集均值平移校正。边界样本量和CPU来源覆盖率须与结果一起解释。", "",
              "## 文件与复跑", "", f"完整步骤（首次运行；已存在结果时可用 --output 指定diagnostics下的新子目录）：`{command}`", "",
              f"仅重画图并重建本结论：`{replot}`", "",
              "- peak_offset_diag.csv：逐个已验收TP的导出残差、区域、信号分位、共同TP身份、CPU原始峰/offset及分解项。",
              "- peak_offset_summary.csv：按模型/种子/全部或共同TP/区域/信号分位/x-y轴汇总，包含空层计数；运行完整步骤生成。",
              "- cpu_gpu_parity.csv、rounding_audit.csv、provenance.json：来源覆盖率、边界舍入事件和输入哈希；运行完整步骤生成。",
              "- peak_offset_overview.png/.svg：固定0.1px分箱残差直方图及H0来源峰位移散点；运行完整步骤或仅重画命令生成。",
              "- peak_offset_snr.png/.svg：共同TP分信号四分位的逐种子均值及三种子均值±样本SD；同上命令生成。",
              "- CONCLUSION.md：本文件；同上命令生成。未开展FP逐条分类、盲审抽样或机制改动。"]
    (out / "CONCLUSION.md").write_bytes(("\n".join(lines)+"\n").encode("utf-8"))
