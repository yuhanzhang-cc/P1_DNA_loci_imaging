# 五模型柱状图、配对折线与显著性说明

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

| 比较 | 平均ΔF1（百分点） | 原始p | Holm校正p | 标记 |
|---|---:|---:|---:|---|
| R1-G0 | -0.503 | 0.250 | 1.000 | ns |
| R1-Qglobal | -0.137 | 0.750 | 1.000 | ns |
| G0-R0 | +1.858 | 0.250 | 1.000 | ns |
| Qglobal-R0 | +1.492 | 0.250 | 1.000 | ns |
| R1-S1 | +0.200 | 0.750 | 1.000 | ns |

图像说明：五模型开发F1接近0.81–0.83，G0平均最高；配对线显示种子差异。主张仍须区分描述提升、机制证据和独立泛化。

复现：使用现有独立`.venv/spotiflow_cpu`运行`python deepblink_pytorch/scripts/plot_closing_comparison.py`；重复输出须显式`--force`或指定新目录。
来源表和脚本哈希、环境版本及输出文件哈希见`figure_manifest.json`；精确数值见`exploratory_seed_statistics.csv`。
本次不安装依赖、不重训模型、不读取测试数组。

方法与软件参考：
- SciPy `permutation_test` paired samples: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
- Kassis T., Agarwal V., He Y., Patel D., Brueckner A. M. (2026). Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065
  本次采用scientific-visualization和statistical-analysis技能的图形完整性及统计报告流程；该软件引用不作为模型有效性的证据。
