"""Read-only closing-result figures and explicitly exploratory paired-seed tests."""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import scipy
from scipy.stats import permutation_test

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/next_stage_v3_closing_controls"
ORDER = ("R0", "S1", "R1", "G0", "Qglobal")
LABELS = ("R0", "S1", "R1", "G0", "Q-global")
SEEDS = (42, 43, 44)
CONTRASTS = (("R1", "G0"), ("R1", "Qglobal"), ("G0", "R0"), ("Qglobal", "R0"), ("R1", "S1"))
COLORS = ("#666666", "#A66B00", "#0072B2", "#007F60", "#9B5286")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Require a nonempty finite p-value vector in [0,1].")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def stars(p):
    if p < .0001:
        return "****"
    if p < .001:
        return "***"
    if p < .01:
        return "**"
    if p < .05:
        return "*"
    return "ns"


def exact_paired_signflip(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.shape != (3,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Require exactly three aligned, finite seed pairs.")
    differences = a - b
    null = np.asarray(list(product((-1., 1.), repeat=3))) @ differences / 3
    observed = float(np.mean(differences))
    tolerance = 1e-12
    p = float(np.mean(np.abs(null) >= abs(observed) - tolerance))
    # Independent implementation check against SciPy's exact paired-label swaps.
    reference = permutation_test((a, b), lambda x, y: np.mean(x-y),
                                 permutation_type="samples", vectorized=False,
                                 n_resamples=np.inf, alternative="two-sided")
    if not np.isclose(p, reference.pvalue, atol=1e-12):
        raise AssertionError("Enumeration and SciPy exact test disagree.")
    return p


def self_check():
    assert exact_paired_signflip([1., 2., 3.], [0., 0., 0.]) == .25
    assert exact_paired_signflip([0., 0., 0.], [0., 0., 0.]) == 1.
    assert np.allclose(holm_adjust([.01, .04, .03]), [.03, .06, .06])
    assert [stars(p) for p in [.05, .049, .009, .0009, .00009]] == ["ns", "*", "**", "***", "****"]


def load_scores():
    checks = json.loads((SOURCE / "delivery_checks.json").read_text(encoding="utf-8"))
    if checks["status"] != "complete" or checks["training_tasks"] != 6 or checks["test_arrays_loaded"]:
        raise ValueError("Figures require completed development-only results.")
    with (SOURCE / "validation_comparison.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    scores = {}
    for row in records:
        if row["model"] not in ORDER:
            continue
        key = row["model"], int(row["seed"])
        if key in scores:
            raise ValueError("Duplicate model/seed.")
        scores[key] = float(row["f1"])
    if set(scores) != {(m, s) for m in ORDER for s in SEEDS}:
        raise ValueError("Missing/unexpected model or seed.")
    values = np.array([[scores[m, s] for m in ORDER] for s in SEEDS])
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("F1 must be finite in [0,1].")
    return scores, values


def bracket(ax, x1, x2, y, label):
    ax.plot([x1, x1, x2, x2], [y, y+.009, y+.009, y], c="#222222", lw=.9)
    ax.text((x1+x2)/2, y+.013, label, ha="center", va="bottom", fontsize=9)


def decorate(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E3E3E3", linewidth=.7)
    ax.tick_params(labelsize=10)


def export(fig, output, stem, artifacts):
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white", transparent=False)
        artifacts.append(dict(path=path.name, sha256=digest(path)))
    plt.close(fig)


def plot_figures(values, tests, output):
    means, sd = values.mean(0), values.std(0, ddof=1)
    x = np.arange(len(ORDER))
    artifacts = []
    style = {"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
             "axes.titleweight": "bold", "axes.labelsize": 12, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10.4, 6.6))
        fig.subplots_adjust(left=.09, right=.97, bottom=.22, top=.84)
        ax.bar(x, means, yerr=sd, capsize=4, width=.62, color=COLORS,
               edgecolor="#333333", linewidth=.7, error_kw={"elinewidth": 1.2})
        for i, (seed, marker) in enumerate(zip(SEEDS, ("o", "s", "^"))):
            ax.plot(x + (i-1)*.13, values[i], linestyle="none", marker=marker,
                    markerfacecolor="white", markeredgecolor="#202020", markersize=5.5,
                    label=f"Seed {seed}")
        for index, mean in enumerate(means):
            ax.text(index, mean+sd[index]+.023, f"{mean:.4f}", ha="center", fontsize=10)
        bracket(ax, 2, 3, .92, f"{tests[0]['annotation']} (Holm p={tests[0]['p_holm']:.3f})")
        bracket(ax, 2, 4, 1.005, f"{tests[1]['annotation']} (Holm p={tests[1]['p_holm']:.3f})")
        ax.set(xticks=x, xticklabels=LABELS, ylabel="Development micro-F1", ylim=(0, 1.12))
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.legend(loc="upper left", bbox_to_anchor=(-.005, 1.01), frameon=False, fontsize=9)
        decorate(ax)
        fig.suptitle("Model comparison: means with all three seed values", y=.96, fontsize=15, weight="bold")
        fig.text(.09, .12, "Bars: mean ± sample SD; symbols: paired training seeds 42/43/44 (n=3).", fontsize=10)
        fig.text(.09, .075, "Brackets: the two primary contrasts. All five contrasts corrected together with Holm.", fontsize=9)
        fig.text(.09, .035, "Exploratory exact paired sign-flip tests; validation-selected results, not independent testing.", fontsize=9)
        export(fig, output, "model_f1_bars", artifacts)

        fig, ax = plt.subplots(figsize=(10.4, 5.7))
        fig.subplots_adjust(left=.1, right=.97, bottom=.24, top=.82)
        for index, (seed, marker, linestyle, color) in enumerate(zip(
                SEEDS, ("o", "s", "^"), ("--", ":", "-."), ("#0072B2", "#A66B00", "#007F60"))):
            ax.plot(x, values[index], marker=marker, linestyle=linestyle, color=color,
                    linewidth=1.4, markersize=6, label=f"Seed {seed}")
        ax.errorbar(x, means, yerr=sd, color="#252525", marker="D", linestyle="-",
                    linewidth=2.2, capsize=4, label="Mean ± SD", zorder=5)
        low = np.floor((min(values.min(), (means-sd).min())-.005)*100)/100
        high = np.ceil((max(values.max(), (means+sd).max())+.005)*100)/100
        ax.set(xticks=x, xticklabels=LABELS, ylabel="Development micro-F1 (zoomed axis)", ylim=(low, high))
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=4, frameon=False, fontsize=10)
        decorate(ax)
        fig.suptitle("Paired seed profiles reveal model-to-model differences", y=.96, fontsize=15, weight="bold")
        fig.text(.1, .13, "Each thin line connects the same training seed; model categories are not a time series.", fontsize=10)
        fig.text(.1, .075, "Thick black line: mean ± sample SD. The nonzero y-axis is explicitly zoomed.", fontsize=10)
        fig.text(.1, .03, "Shared validation set and 20 × 31 model selection; n=3 seeds, not 66 independent images.", fontsize=9)
        export(fig, output, "model_f1_paired_lines", artifacts)

        fig, ax = plt.subplots(figsize=(10.4, 6.2))
        fig.subplots_adjust(left=.1, right=.97, bottom=.23, top=.83)
        differences = np.array([[r[f"delta_seed_{seed}"]*100 for r in tests] for seed in SEEDS])
        delta_mean, delta_sd = differences.mean(0), differences.std(0, ddof=1)
        ax.bar(x, delta_mean, yerr=delta_sd, color="#718794", edgecolor="#303030",
               capsize=4, width=.6)
        for index, (seed, marker) in enumerate(zip(SEEDS, ("o", "s", "^"))):
            ax.plot(x + (index-1)*.13, differences[index], linestyle="none", marker=marker,
                    markerfacecolor="white", markeredgecolor="#202020", markersize=5.5,
                    label=f"Seed {seed}")
        for index, row in enumerate(tests):
            top = max(0., delta_mean[index]+delta_sd[index], differences[:, index].max())
            ax.text(index, top+.24, f"{row['annotation']}\nadj. p={row['p_holm']:.3f}",
                    ha="center", fontsize=9, linespacing=1.5)
        lower = min(-.4, float(min(differences.min(), (delta_mean-delta_sd).min()))-.55)
        upper = max(float(max(differences.max(), (delta_mean+delta_sd).max()))+.95, 1.)
        ax.axhline(0, color="#333333", lw=1)
        ax.set(xticks=x, xticklabels=[r["comparison"].replace("Qglobal", "Q-global") for r in tests],
               ylabel="Paired F1 difference (percentage points)", ylim=(lower, upper))
        ax.legend(loc="lower left", bbox_to_anchor=(0., 1.005), ncol=3, frameon=False, fontsize=9)
        decorate(ax)
        fig.suptitle("Effect sizes and corrected significance for every planned contrast", y=.96,
                     fontsize=14, weight="bold")
        fig.text(.1, .13, "Positive values favor the first named model. Bars: paired mean difference ± sample SD.", fontsize=10)
        fig.text(.1, .075, "Two-sided exact sign flips over 3 seed pairs; Holm correction across all 5 contrasts.", fontsize=10)
        fig.text(.1, .03, "* p<0.05; ** p<0.01; *** p<0.001; **** p<0.0001 (adjusted p). ns does not mean equivalence.", fontsize=9)
        export(fig, output, "paired_f1_differences", artifacts)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SOURCE / "03_figures_barline_20261028")
    parser.add_argument("--force", action="store_true", help="Overwrite only this script's named figure/report outputs")
    args = parser.parse_args()
    self_check()
    if args.output.exists() and not args.force:
        raise FileExistsError("Choose a new output directory, or use --force after reviewing it.")
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = [SOURCE / n for n in ("delivery_checks.json", "validation_comparison.csv", "paired_summary.json")]
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_files}
    # Written before numerical testing. Tests requested after results were known;
    # they are explicitly exploratory, not retrospective preregistration.
    plan = dict(role="post_hoc_exploratory_seed_variability_analysis", endpoint="validation micro-F1",
                sampling_unit="training seed, paired within the same frozen development dataset",
                seeds=list(SEEDS), comparisons=[f"{a}-{b}" for a, b in CONTRASTS],
                method="exact_two_sided_paired_signflip_of_mean_difference", permutations=8,
                multiplicity="Holm familywise correction across all five contrasts", alpha=.05,
                data_selection="20 checkpoints x 31 thresholds per run; this optimism is not corrected by these tests",
                independence="three stochastic seeds are not biological replicates; image source independence unverified",
                assumptions="paired differences independent across seeds; null sign symmetry/within-pair exchangeability; unassessable with n=3",
                exclusions="none among the five displayed models and three seeds",
                scientific_status="not confirmatory and not independent generalization evidence")
    write_json(args.output / "statistical_plan.json", plan)
    scores, values = load_scores()
    tests = []
    for a, b in CONTRASTS:
        differences = np.array([scores[a, seed]-scores[b, seed] for seed in SEEDS])
        tests.append(dict(comparison=f"{a}-{b}", n_seed_pairs=3, mean_delta_f1=float(differences.mean()),
                          sample_sd_delta=float(differences.std(ddof=1)),
                          **{f"delta_seed_{s}": float(d) for s, d in zip(SEEDS, differences)},
                          p_raw=exact_paired_signflip([scores[a, s] for s in SEEDS], [scores[b, s] for s in SEEDS])))
    for row, adjusted in zip(tests, holm_adjust([r["p_raw"] for r in tests])):
        row.update(p_holm=float(adjusted), annotation=stars(adjusted))
    write_csv(args.output / "exploratory_seed_statistics.csv", tests)
    write_csv(args.output / "source_seed_scores.csv", [dict(model=m, seed=s, f1=scores[m, s]) for m in ORDER for s in SEEDS])
    write_csv(args.output / "model_summary.csv", [dict(model=m, n_seeds=3, mean_f1=float(values[:, i].mean()),
              sample_sd_f1=float(values[:, i].std(ddof=1))) for i, m in enumerate(ORDER)])
    artifacts = plot_figures(values, tests, args.output)
    for p in source_files:
        if digest(p) != hashes[p.relative_to(ROOT).as_posix()]:
            raise AssertionError("A protected source table changed.")
    write_json(args.output / "figure_manifest.json", dict(source_sha256=hashes, script_sha256=digest(__file__),
               artifacts=artifacts, numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__,
               python=platform.python_version(), test=plan, internal_self_checks_passed=True,
               original_training_files_modified=False, test_arrays_loaded=False))
    rows = ["| 比较 | 平均ΔF1（百分点） | 原始p | Holm校正p | 标记 |", "|---|---:|---:|---:|---|"]
    rows += [f"| {r['comparison']} | {100*r['mean_delta_f1']:+.3f} | {r['p_raw']:.3f} | {r['p_holm']:.3f} | {r['annotation']} |" for r in tests]
    readme = """# 五模型柱状图、配对折线与显著性说明

本包新增图件，不覆盖原训练曲线、冻结协议或完成报告。

- `model_f1_bars.png/.svg`：均值±样本SD柱状图，零基线；保留三个种子的原始值，括号展示两个主要比较。
- `model_f1_paired_lines.png/.svg`：同一种子在模型之间的配对折线及均值±SD，使用明确标注的局部放大纵轴。类别没有时间或连续趋势含义。
- `paired_f1_differences.png/.svg`：全部五项配对差、误差和校正显著性，不隐藏反向或不显著结果。

统计单位是三个训练种子，不是22张验证图、66个图像×种子组合，也不是372个点。
检验为双侧精确配对符号翻转，枚举全部2^3=8种符号，并与SciPy精确配对置换核对。
五项既定对比一起作Holm校正。推断依赖种子间独立、零假设下配对差的符号对称/标签交换性；n=3无法检验这些假设。
这些p值是在看到成绩后按本次要求添加的探索性分析，不追认成预注册检验。
原验证集已用于每次20×31选模，校正五项对比不消除这种选择偏差。

星号规则使用**校正后p**：* <0.05；** <0.01；*** <0.001；**** <0.0001；其他为ns。
三个种子的这种双侧精确检验，最小原始p为2/8=0.25，因此当前无法获得星号。
不改用图像/点作为独立重复，也不为得到星号换检验。ns不是无效或等效的证明。
效应差采用原始F1及百分点，SD只描述种子波动，不冒充生物学置信区间。

""" + "\n".join(rows) + """

图像说明：五模型开发F1接近0.81–0.83，G0平均最高；配对线显示种子差异。主张仍须区分描述提升、机制证据和独立泛化。

复现：使用现有独立`.venv/spotiflow_cpu`运行`python deepblink_pytorch/scripts/plot_closing_comparison.py`；重复输出须显式`--force`或指定新目录。
来源表和脚本哈希、环境版本及输出文件哈希见`figure_manifest.json`；精确数值见`exploratory_seed_statistics.csv`。
本次不安装依赖、不重训模型、不读取测试数组。

方法与软件参考：
- SciPy `permutation_test` paired samples: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
- Kassis T., Agarwal V., He Y., Patel D., Brueckner A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
  本次采用scientific-visualization和statistical-analysis技能的图形完整性及统计报告流程；该软件引用不作为模型有效性的证据。
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), contrasts=tests, figures=artifacts), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
