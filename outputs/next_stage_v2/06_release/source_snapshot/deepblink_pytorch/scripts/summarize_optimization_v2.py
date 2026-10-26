"""Summarize the completed first screen without any test-set inference."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from optimize_v2 import OUT, ROOT, sha256, write_csv, write_json, load_development, decode_grid
from deepblink_torch.evaluation import match_coordinates


def size_summary(name, path):
    with path.open() as handle:
        images = list(csv.DictReader(handle))
    rows = []
    for size in (256, 512):
        selected = [r for r in images if int(r["height"]) == size]
        tp, fp, fn = [sum(int(float(r[k])) for r in selected) for k in ("tp", "fp", "fn")]
        rows.append(dict(model=name, original_height=size, n_images=len(selected),
                         tp=tp, fp=fp, fn=fn, precision=tp/(tp+fp) if tp+fp else 0.,
                         recall=tp/(tp+fn) if tp+fn else 0.,
                         f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.))
    return rows


def localization_summary(folder, development):
    val = json.loads((folder / "validation_summary.json").read_text())
    with np.load(folder / "prediction_grids.npz") as archive:
        assert list(archive["sample_id"]) == list(development["sample_id"])
        grids = archive["grids"]
    distances, predictions, matches = [], [], []
    for sid, grid, shape, truth in zip(development["sample_id"], grids, development["original_shape"], development["y"]):
        coords, scores = decode_grid(grid, shape, val["threshold"])
        match = match_coordinates(coords, truth, 3.)
        distances.extend(match.matched_distances.tolist())
        predictions.extend(dict(sample_id=str(sid), pred_index=i, y_px=float(y), x_px=float(x), score=float(scores[i]))
                           for i, (y, x) in enumerate(coords))
        def append_match(status, pi=None, ti=None, distance=None):
            matches.append(dict(sample_id=str(sid), status=status, pred_index=pi, reference_index=ti,
                                pred_y_px=float(coords[pi, 0]) if pi is not None else None,
                                pred_x_px=float(coords[pi, 1]) if pi is not None else None,
                                reference_y_px=float(truth[ti, 0]) if ti is not None else None,
                                reference_x_px=float(truth[ti, 1]) if ti is not None else None,
                                score=float(scores[pi]) if pi is not None else None, distance_px=distance))
        for pi, ti, distance in zip(match.matched_pred_indices, match.matched_true_indices, match.matched_distances):
            append_match("TP", int(pi), int(ti), float(distance))
        for pi in match.unmatched_pred_indices:
            append_match("FP", int(pi))
        for ti in match.unmatched_true_indices:
            append_match("FN", ti=int(ti))
    assert len(distances) == val["tp"]
    assert sum(row["status"] == "FP" for row in matches) == val["fp"]
    assert sum(row["status"] == "FN" for row in matches) == val["fn"]
    write_csv(folder / "prediction_points.csv", predictions)
    write_csv(folder / "matches.csv", matches)
    return dict(median_error_px=float(np.median(distances)) if distances else None,
                mean_error_px=float(np.mean(distances)) if distances else None,
                rmse_error_px=float(np.sqrt(np.mean(np.square(distances)))) if distances else None)


def main():
    development = load_development("valid")
    reference_dir = OUT / "00_protocol/reference_seed_42"
    reference = json.loads((reference_dir / "validation_summary.json").read_text())
    reference_mechanism = json.loads((OUT / "08_innovation/development_audit/error_summary.json").read_text())
    rows = [dict(model="deepblink", role="reference", seed=42, best_epoch=94,
                 parameters=1258723, **{k: reference[k] for k in
                 ("f1", "precision", "recall", "threshold", "tp", "fp", "fn", "count_mae", "macro_f1")},
                 delta_f1_pp=0., **{k: reference_mechanism[k] for k in
                 ("weak_truth_recall", "off_peak_fp_count", "high_confidence_fp_count")},
                 **localization_summary(reference_dir, development))]
    registry = []
    size_rows = size_summary("deepblink", reference_dir / "per_image_metrics.csv")
    roles = {"deeploci_evidence": "core_hypothesis", "deeploci_local": "simple_alternative",
             "deeploci_feature_evidence": "core_hypothesis_revision", "deepblink_masked_se": "pooling_diagnostic"}
    for name in roles:
        folder = OUT / "02_architecture" / name / "seed_42"
        status = json.loads((folder / "status.json").read_text())
        if status["status"] != "complete":
            raise RuntimeError(f"Screen incomplete: {name}")
        with (folder / "training_metrics.csv").open() as handle:
            history = list(csv.DictReader(handle))
        assert len(history) == 100
        assert all(np.isfinite(float(row[key])) for row in history for key in ("train_loss", "val_loss"))
        checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=False)
        assert checkpoint["best_val_loss"] == min(float(row["val_loss"]) for row in history)
        assert all(bool(torch.isfinite(value).all()) for value in checkpoint["model_state_dict"].values())
        config_path = folder / "registered_config.json"
        config_hash = sha256(config_path)
        config_path.with_suffix(".sha256").write_text(config_hash, encoding="utf-8")
        audit = folder / "development_diagnostics"
        if not (audit / "error_summary.json").exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name("diagnose_v2.py")),
                            "--predictions", str(folder / "validation_v2"), "--output", str(audit)], check=True)
        val = json.loads((folder / "validation_v2/validation_summary.json").read_text())
        size_rows.extend(size_summary(name, folder / "validation_v2/per_image_metrics.csv"))
        mech = json.loads((audit / "error_summary.json").read_text())
        role = roles[name]
        rows.append(dict(model=name, role=role, seed=42, best_epoch=checkpoint["epoch"],
                         parameters=checkpoint["parameter_count"], **{k: val[k] for k in
                         ("f1", "precision", "recall", "threshold", "tp", "fp", "fn", "count_mae", "macro_f1")},
                         delta_f1_pp=100*(val["f1"]-reference["f1"]), **{k: mech[k] for k in
                         ("weak_truth_recall", "off_peak_fp_count", "high_confidence_fp_count")},
                         **localization_summary(folder / "validation_v2", development)))
        registry.append(dict(model=name, role=role, seed=42, epochs=100,
                             optimizer_updates=2200, status="complete", config_sha256=config_hash,
                             checkpoint_sha256=sha256(folder / "best_model.pt"),
                             validation_f1=val["f1"], test_evaluated=False,
                             wall_seconds=status["elapsed_seconds"]))
    write_csv(OUT / "validation_comparison.csv", rows)
    write_csv(OUT / "trial_registry.csv", registry)
    write_csv(OUT / "size_validation_metrics.csv", size_rows)
    write_json(OUT / "validation_comparison.json", rows)
    core = max([r for r in rows if r["role"] in ("core_hypothesis", "core_hypothesis_revision")], key=lambda r:r["f1"])
    write_json(OUT / "stage_status.json", dict(protocol_id="pooled220_methods_v2",
               stage="initial_P0_P1_P2_screen", full_training_trials_completed=len(registry),
               architecture_slots_used=len(registry), architecture_slots_max=4,
               core_single_seed_validation_gain=core["delta_f1_pp"] > 0,
               structural_contribution_established=False, independent_validation=False,
               test_evaluated=False, blind_annotation_review="pending"))
    evidence = [dict(contribution="data_expansion", status="legacy_internal_support",
                    evidence="+12.77 pp historical matched-control mean F1", limitation="not architecture gain"),
                dict(contribution="evaluation", status="implementation_verified",
                    evidence="maximum-cardinality point matcher and validation-only v2 calibration",
                    limitation="no architecture claim"),
                dict(contribution="structure_core_candidates", status="single_seed_development_screen_complete",
                    evidence=f"best core {core['model']} validation F1 {core['f1']:.6f}; delta {core['delta_f1_pp']:+.3f} pp",
                    limitation="repeatability and structural attribution and independent validation pending")]
    write_csv(OUT / "06_release/innovation_evidence.csv", evidence)
    lines = ["# 首轮优化阶段汇报", "", "范围：P0协议与参考、P1问题和设计审查、P2的D/B单种子结构筛查。",
             "同一220对数据、176训练/22验证，T_ref和seed42；本轮未评估测试集。所有表中分数均为验证集点F1，不是测试成绩。", "",
             "| 模型 | 参数 | 最佳轮 | P | R | F1 | 对基线变化(pp) | TP/FP/FN | 弱点召回 | 离峰FP | 高置信FP |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    lines[2] = "范围：P0协议与参考、P1问题和设计审查、P2的A/B/D/D2四项单种子结构筛查。"
    winner = max(rows, key=lambda row: row["f1"])
    lines[4:4] = ["", f"本轮最高验证点F1为{winner['f1']:.6f}，来自{winner['model']}。这不是独立测试结论，也不证明全局最优。"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['parameters']:,} | {row['best_epoch']} | {row['precision']:.4f} | "
                     f"{row['recall']:.4f} | {row['f1']:.4f} | {row['delta_f1_pp']:+.2f} | "
                     f"{row['tp']}/{row['fp']}/{row['fn']} | {row['weak_truth_recall']:.4f} | "
                     f"{row['off_peak_fp_count']} | {row['high_confidence_fp_count']} |")
    lines += ["", "D为预测位置条件化的原图统计证据融合；D2改为可学习局部特征对比。B为简单3×3占据特征头；A仅修正SE池化中的无效区域统计。四个候选都从头训练100轮，未增加损失或标签。",
              "", "原主干三种子已按v2重新校准，但此处对照只取同seed42；不把旧v1校准分数与新结构分数直接相减。原模型三个验证F1为0.828229/0.807895/0.822888。",
              "", "机制指标仅为预先固定的图像代理：弱点阈值来自训练真值局部z的下四分位，离峰FP为距局部最亮像素>3px，绝对置信度>0.9的FP另列；这些代理不能证明真实背景成因。",
              "", "匹配修复解决了拥挤排列的少配反例；132个旧权重验证图/阈值比较中TP未改变。keep-all端点明确包含数值为0的sigmoid输出。历史模型和报告保持原样。",
              "", "本轮完成4/4个结构筛查槽位。单个seed、反复使用的22张验证图不能支持可靠架构改进结论；核心结构贡献尚未建立。三种子重复、容量/固定中心等从头重训控制、强基线及独立新采集数据仍需后续scope完成。",
              "", "40张开发图的盲法人工标注复核队列已备好，复核未完成。部分注释相对FP具有点状外观，不能将其自动重标或都认定为背景。",
              "", "数据、配置、源码、环境、逐轮损失、最优checkpoint、训练曲线、31阈值验证表、逐图指标、错误表和patch图均有保存。旧220/89/113输出未覆盖；本轮没有更改共享环境或安装依赖。"]
    lines += ["", "A选中的阈值0.025位于预设线性网格的最低内部点；原图得分覆盖已保存，低分区间可能需要下一版协议中共同预注册的细化，不能把本轮网格内最优解释为全局阈值最优。未追加有利于某候选的阈值搜索。",
              "", "逐点导出见各validation_v2目录（参考模型在00_protocol/reference_seed_42）：prediction_points.csv和matches.csv。定位误差是TP条件下统计，不能忽略FN直接宣称定位整体更好。",
              "", "解读与下一步见[结果解读与下一步.md](结果解读与下一步.md)；初步资源测量见[运行成本](02_architecture/runtime_benchmark.csv)，验收记录见[delivery_checks.json](delivery_checks.json)。"]
    (OUT / "阶段汇报.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
