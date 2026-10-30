"""Post-hoc validation accounting; no model fitting or test-array access."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from error_context_diagnostics import _extract_patch_features
from deepblink_torch.evaluation import match_coordinates
from deepblink_torch.round2_protocol import THRESHOLDS_V3, decode_candidates, sweep_candidate_counts

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/next_stage_v3"
ARCHIVE = ROOT / "outputs/261024_clean220_random/dna_clean220_random.npz"
WEAK_CUTOFF = 2.4566830772038957


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, values):
    if not values: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def write_interpretation(completed, comparisons, inherited, size_rows, radius_rows, output_root=OUT):
    """Explain completed evidence without feeding post-hoc findings into training."""
    old = {r["seed"]: r for r in inherited}
    lines = ["# 第二轮开发结果解释", "",
             "本文仅使用已完成trial的验证预测；运行状态以../controller_status.json为准。所有分数来自同一22张开发验证图，不能作为新的测试成绩或独立泛化证据。",
             "", "每次训练按预定20个checkpoint×31阈值选模，存在验证选择偏差。旧权重采用其历史原生损失checkpoint和本轮31阈值；它不是同训练顺序、同选模机会的结构对照，但保留它可以防止把超过一次较弱的新R0误称为整体进步。",
             "", "| 模型 | seed | F1 | 弱点检出/96 | FP | FN | 相对新R0 ΔF1 | 相对同seed旧权重 ΔF1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for (key, seed), (_, result) in sorted(completed.items()):
        base = completed.get(("R0", seed))
        delta = f"{result['f1']-base[1]['f1']:+.4f}" if base else "待R0完成"
        lines.append(f"| {key} | {seed} | {result['f1']:.4f} | {result['weak_tp']}/96 | {result['fp']} | {result['fn']} | {delta} | {result['f1']-old[seed]['f1']:+.4f} |")
    lines += ["", "## 是否提升", ""]
    routes = [key for key in ("R1", "R2") if (key, 42) in completed and ("R0", 42) in completed]
    if not routes:
        lines.append("核心候选尚无完整配对结果，暂不能判断架构是否提升。R0自己的新旧差异不属于结构改进。")
    for key in routes:
        result, base = completed[key, 42][1], completed["R0", 42][1]
        gain = result["f1"]-base["f1"]
        weak_delta = result["weak_recall"]-base["weak_recall"]
        passed = gain >= .005 and weak_delta >= -.02
        lines.append(f"{key} seed42相对新R0：ΔF1={gain:+.4f}，弱点Recall变化{weak_delta:+.4f}；{'通过' if passed else '未通过'}预定筛查门槛。")
        if result["f1"] <= old[42]["f1"]:
            lines.append(f"其F1仍未超过同seed旧权重参照{old[42]['f1']:.4f}；即使通过新R0筛查，也不能据此宣称整体性能已超过已有220模型。")
        else:
            lines.append(f"其开发F1高于同seed旧权重参照{old[42]['f1']:.4f}，但差异仍包含采样及选模变化；结构归因以新R0配对比较为主。")
        seeds = [seed for seed in (42, 43, 44) if (key, seed) in completed and ("R0", seed) in completed]
        if len(seeds) == 3:
            gains = [completed[key, seed][1]["f1"]-completed["R0", seed][1]["f1"] for seed in seeds]
            lines.append(f"三个配对种子的ΔF1依次为{', '.join(f'{v:+.4f}' for v in gains)}；平均{statistics.mean(gains):+.4f}，样本SD {statistics.stdev(gains):.4f}。其中{sum(v > 0 for v in gains)}/3为正，仅作开发稳定性描述，不是显著性检验。")
        else:
            lines.append("尚无完整三个配对种子，单次筛查结果不能称为稳定收益。")
        lines.append("")
    lines += ["## 收益或损失体现在哪里", ""]
    if not comparisons:
        lines.append("等待候选完成后再核对逐点变化。")
    for item in comparisons:
        lines.append(f"- {item['model']} seed{item['seed']}相对新R0：新增检出{item['gained_truths']}个、丢失{item['lost_truths']}个参考点；其中弱点新增{item['gained_weak_truths']}个、丢失{item['lost_weak_truths']}个。FP变化{item['fp_delta']:+d}，FN变化{item['fn_delta']:+d}。")
    lines += ["", "这些计数说明性能变化的构成。参考标注尚未独立复核，FP/FN不能直接解释为生物学假点/真点；这也不是对网络内部原因的因果证明。", "",
              "## 核心算子和简单对照", ""]
    controls_seen = False
    for route, control in (("R1", "S1"), ("R2", "S2")):
        for seed in (42, 43, 44):
            if (route, seed) not in completed or (control, seed) not in completed:
                continue
            controls_seen = True
            full, simple = completed[route, seed][1], completed[control, seed][1]
            delta = full["f1"]-simple["f1"]
            lines.append(f"- seed{seed}：{route}−{control}的ΔF1={delta:+.4f}，弱点检出差{full['weak_tp']-simple['weak_tp']:+d}/96，FP差{full['fp']-simple['fp']:+d}。")
            if delta <= 0:
                lines.append(f"  该次结果不支持复杂选择算子优于{control}；不能将输入分支或容量的收益写成选择机制的贡献。")
    if not controls_seen:
        lines.append("完整路线及对应简单控制尚未同时完成，暂不作机制结论。")
    lines += ["", "相同FP预算的注册工作点见fixed_fp_budget.csv，只在各模型已选checkpoint的原31阈值中选择，没有额外阈值搜索。", "",
              "## 尺寸与定位核查", ""]
    for key in sorted({key for key, seed in completed if seed == 42}):
        sizes = {r["size"]: r for r in size_rows if r["model"] == key and r["seed"] == 42}
        radius = {r["radius_px"]: r for r in radius_rows if r["model"] == key and r["seed"] == 42}
        lines.append(f"- {key} seed42：256图F1 {sizes[256]['f1']:.4f}、512图F1 {sizes[512]['f1']:.4f}；固定选定阈值，匹配半径1/2/3px的F1依次{radius[1.]['f1']:.4f}/{radius[2.]['f1']:.4f}/{radius[3.]['f1']:.4f}。")
    lines += ["", "尺寸与来源、标定及成像内容可能同时变化，以上为描述分层。定位误差只针对已匹配TP，不能用其改善替代检测性能。", "",
              "## 下一步的取舍", "",
              "1. 只有完整路线超过新R0且优于对应简单控制，才继续讨论选择算子的价值；若简单控制持平或更好，保留更简单结构，停止把复杂选择本身作为有效创新的主张。",
              "2. 只有H0改善时，应将结论写成高分辨率表示、监督与结构共同变化的结果；热图输出和offset已有成熟先例，不能单靠本数据集训练构成创新。H0若失败，也只淘汰本轮固定配方，不代表所有密集方法都无效。",
              "3. 若FP减少伴随弱点大量丢失，沿用第一轮教训，停止继续叠加抑制分支。下一版需另行登记正负证据/定位的耦合诊断及有针对性的监督设计，并重新训练配对R0。不得在本轮临时增加loss或参数搜索。",
              "4. 若细节路线显示稳定价值，正式消融应补充普通空间重排/投影替代和参数控制；若上下文路线有价值，应补充普通大核与保留中央位置的替代。它们均在本轮预算之外，不能把单次急性置零当作重新训练消融。",
              "5. 在论文级结论之前，完成盲法标注复核、最近算子与方法差异审查，并准备按生物学来源隔离的独立验证。需要改标注时新建数据版本并重训所有比较方法。"]
    (output_root / "02_analysis/结果解释.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(trial_roots=None, output_root=OUT):
    """Allow separately authorized seed completions without rewriting old reports."""
    trial_roots = [OUT / "01_screen"] if trial_roots is None else list(trial_roots)
    output_root = Path(output_root)
    with np.load(ARCHIVE, allow_pickle=True) as archive:
        # Exact allowed-key access; do not iterate through or extract other arrays.
        ids = archive["sample_id_valid"]
        shapes = archive["original_shape_valid"]
        images = archive["x_valid"]
        truths = archive["y_valid"]
    features = {}
    for sid, shape, image, truth in zip(ids, shapes, images, truths):
        h, w = map(int, shape)
        for i, (y, x) in enumerate(truth):
            item = _extract_patch_features(image[:h, :w], y, x, 10)
            features[(str(sid), i)] = dict(sample_id=str(sid), true_index=i, y=float(y), x=float(x),
                         local_zscore=float(item["local_zscore"]) if np.isfinite(item["local_zscore"]) else None,
                         weak=bool(item["local_zscore"] < WEAK_CUTOFF))
    assert sum(f["weak"] for f in features.values()) == 96
    write_csv(output_root / "02_analysis/validation_truth_features.csv", list(features.values()))
    completed, size_rows, radius_rows, frontiers = {}, [], [], {}
    paths = sorted(p for folder in trial_roots for p in Path(folder).glob("*/seed_*/selection.json"))
    for path in paths:
        trial = path.parent
        config = json.loads((trial / "registered_config.json").read_text(encoding="utf-8"))
        if json.loads((trial / "status.json").read_text(encoding="utf-8"))["status"] != "complete":
            raise ValueError(f"Analysis requires a completed trial: {trial}")
        if (config["key"], config["seed"]) in completed:
            raise ValueError(f"Duplicate model/seed in analysis inputs: {trial}")
        result = json.loads(path.read_text(encoding="utf-8"))["primary"]
        matched = {(r["sample_id"], int(r["true_index"])) for r in rows(trial / "matches.csv")}
        assert len(matched) == result["tp"]
        assert sum(features[k]["weak"] for k in matched) == result["weak_tp"]
        assert len(features)-len(matched) == result["fn"]
        predictions = rows(trial / "prediction_points.csv")
        assert len(predictions)-len(matched) == result["fp"]
        image_metrics = rows(trial / "per_image_metrics.csv")
        for size in (256, 512):
            group = [r for r in image_metrics if int(r["height"]) == size]
            tp, fp, fn = [sum(int(float(r[k])) for r in group) for k in ("tp", "fp", "fn")]
            size_rows.append(dict(model=config["key"], seed=config["seed"], size=size, images=len(group),
                                  tp=tp, fp=fp, fn=fn, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.))
        for radius in (1., 2., 3.):
            counts = np.zeros(3, dtype=int)
            for sid, truth in zip(ids, truths):
                points = np.array([[float(r["y"]), float(r["x"])] for r in predictions if r["sample_id"] == sid], dtype=np.float32).reshape(-1, 2)
                match = match_coordinates(points, truth, radius)
                counts += [match.tp, match.fp, match.fn]
            tp, fp, fn = map(int, counts)
            radius_rows.append(dict(model=config["key"], seed=config["seed"], radius_px=radius,
                                    tp=tp, fp=fp, fn=fn, f1=2*tp/(2*tp+fp+fn), threshold_retuned=False))
        completed[(config["key"], config["seed"])] = (matched, result)
        frontier_path = trial / f"epoch_{result['epoch']:03d}/threshold_metrics.json"
        frontiers[config["key"], config["seed"]] = json.loads(frontier_path.read_text(encoding="utf-8"))
    comparisons = []
    for (key, seed), (matched, result) in completed.items():
        if key == "R0" or ("R0", seed) not in completed: continue
        base, baseline = completed[("R0", seed)]
        gained, lost = matched-base, base-matched
        assert len(gained)-len(lost) == result["tp"]-baseline["tp"]
        details = []
        for k in sorted(features):
            details.append(dict(**features[k], baseline_detected=k in base, candidate_detected=k in matched,
                                change="gained" if k in gained else "lost" if k in lost else "both" if k in matched else "neither"))
        write_csv(output_root / f"02_analysis/{key}/seed_{seed}/truth_detection_changes.csv", details)
        comparisons.append(dict(model=key, seed=seed, delta_f1=result["f1"]-baseline["f1"],
                 gained_truths=len(gained), lost_truths=len(lost),
                 gained_weak_truths=sum(features[k]["weak"] for k in gained),
                 lost_weak_truths=sum(features[k]["weak"] for k in lost),
                 fp_delta=result["fp"]-baseline["fp"], fn_delta=result["fn"]-baseline["fn"]))
    write_csv(output_root / "02_analysis/paired_detection_changes.csv", comparisons)
    write_csv(output_root / "02_analysis/size_validation_metrics.csv", size_rows)
    write_csv(output_root / "02_analysis/radius_sensitivity.csv", radius_rows)
    fixed_fp_rows = []
    for (key, seed), frontier in sorted(frontiers.items()):
        if ("R0", seed) not in completed:
            continue
        budget = completed["R0", seed][1]["fp"]
        point = max((r for r in frontier if r["fp"] <= budget),
                    key=lambda r: (r["tp"], r["precision"], r["threshold"]))
        fixed_fp_rows.append(dict(model=key, seed=seed, reference_fp_budget=budget, **point))
    write_csv(output_root / "02_analysis/fixed_fp_budget.csv", fixed_fp_rows)
    # Frozen historical weights are an additional sanity reference, not a
    # replacement for the independently sampled, prospectively paired v3 R0.
    inherited, inherited_sweeps = [], []
    weak_arrays = [np.array([features[str(sid), i]["weak"] for i in range(len(truth))])
                   for sid, truth in zip(ids, truths)]
    for seed in (42, 43, 44):
        source = ROOT / f"outputs/next_stage_v2/00_protocol/reference_seed_{seed}/prediction_grids.npz"
        with np.load(source, allow_pickle=False) as z:
            assert np.array_equal(ids, z["sample_id"])
            candidates = [decode_candidates(grid, shape) for grid, shape in zip(z["grids"], shapes)]
        counts = [sweep_candidate_counts(c, s, truth) for (c, s), truth in zip(candidates, truths)]
        sweep = []
        for index, threshold in enumerate(THRESHOLDS_V3):
            tp, fp, fn = [sum(c[index][j] for c in counts) for j in range(3)]
            weak_tp = int(sum(np.count_nonzero(w[c[index][3]]) for w, c in zip(weak_arrays, counts)))
            sweep.append(dict(seed=seed, threshold=threshold, tp=tp, fp=fp, fn=fn,
                         precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn),
                         f1=2*tp/(2*tp+fp+fn), weak_tp=weak_tp, weak_recall=weak_tp/96))
        inherited.append(max(sweep, key=lambda r: (r["f1"], r["precision"], r["threshold"])))
        inherited_sweeps.extend(sweep)
    write_csv(output_root / "02_analysis/inherited_weights_new_grid.csv", inherited)
    write_csv(output_root / "02_analysis/inherited_weights_new_grid_all_thresholds.csv", inherited_sweeps)
    write_interpretation(completed, comparisons, inherited, size_rows, radius_rows, output_root)
    metadata = dict(role="post_hoc_validation_accounting_not_causal_mechanism_proof", completed_trials=len(completed),
                 validation_truths=len(features), weak_truths=96, new_training=False, model_inference=False,
                 test_arrays_read=False, annotation_review="pending", source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    metadata["inherited_reference_limit"] = "Frozen old native-loss checkpoints; cached float32 NCHW outputs, same 31 thresholds, one checkpoint per seed. Not matched v3 epoch selection or image order; not a structural attribution contrast."
    metadata["trial_roots"] = [str(Path(p).resolve()) for p in trial_roots]
    path = output_root / "02_analysis/analysis_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(metadata, comparisons=comparisons), indent=2))


if __name__ == "__main__":
    main()
