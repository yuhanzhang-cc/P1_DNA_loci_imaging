# CUDA分析的聚合交付材料

日期：2026-10-30。完整说明见[结果报告](../cuda_analysis_20261030.md)。这里仅包含聚合表、图、方法锁和分析出处；私有逐图/逐点材料与权重保留本地。

| 文件 | 内容 |
|---|---|
| model_summary.csv | 三种子均值/样本SD；Z0/Z1单checkpoint无SD |
| per_seed_metrics.csv | 每次训练的选模、计数、定位、弱点和计数误差 |
| paired_comparisons.csv | 两个探索性对比、精确p、Holm及条件图像bootstrap |
| size_metrics.csv | 256/512尺寸分层，不代表独立泛化 |
| error_complementarity.csv | 相同真值上的检出互补与定位误差 |
| checkpoint_learning.csv | 20个快照的完整图F1与native-loss |
| training_timing.csv | GPU轮次回调/推理时间，不能跨硬件排名 |
| deployment_reload_probe.csv | 3模型×2固定尺寸的CPU重载复核 |
| method_lock_cuda_20261030.json | 三个主选模型、阈值、权重SHA256与训练协议 |
| analysis_provenance.json | 输入哈希、检查项目、环境与解释边界 |
| figure_manifest.json | 图形语义、重复单位、误差棒、导出格式 |

图提供PNG和可编辑文字SVG：`model_comparison`、`paired_seed_comparison`、`training_and_selection`、`precision_recall_diagnostics`。柱图从0开始；误差棒为训练种子样本SD。训练曲线星形仅标出被选epoch，显著性只在配对图明确标`ns`。

复核入口：`deepblink_pytorch/scripts/analyze_spotiflow_cuda.py`。数据为开发验证结果，模型/阈值在同一验证集选取；图像bootstrap未修正选模偏差。当前按一般研究汇报导出，不宣称满足某期刊的投稿规范。
