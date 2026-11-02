# 最短收尾SPEC：先交付，再补最小benchmark，再过核心机制关口

日期：2026-11-02。依据：[六次CUDA结果分析](recipe_cuda_analysis_20261102.md)。目标是得到可复核、可运行、结论清楚的版本，停止无边界的增强/模块搜索。核心创新仍需实证；现有工作不能预写成“我们的最好”。

## A. 已完成的当前交付

- 六次G0配方完整归档与CPU复现；联合增强不升为默认。保留primary/secondary全部结果与负结果。
- 可运行的固定模型推理候选版：Spotiflow适配seed42默认、G0历史简单参考seed42可选；原生TIFF输出像素坐标，阈值/权重绑定，CPU检查通过。该版本承载当前已达到的检测/定位能力。
- 旧113的0.84878单列为历史结果，不进入同条件主表；同22图的旧XPU结果只作带环境标签的背景资料。

## B. 最小GPU补齐：仅R0/H0，共六次

使用完成recipe的同一RTX5090、torch2.11.0+cu128与上传包，176/22划分、100epoch、batch8、原配方无增强、每项2200更新和620次选择均不变。

| 队列 | 数量 | 回答的问题 |
|---|---:|---|
| baseline_r0 | seed42/43/44，3次 | 新CUDA条件下原DeepBlink迁移主干与G0的差异；避免旧113训练重叠 |
| baseline_h0 | seed42/43/44，3次 | dense heatmap+offset表示是否改善检出/定位；损失表示不同需明示 |

最小补齐包位于`outputs/next_stage_v4_recipe_20261101/11_minimum_baseline_handoff/p1_minimum_baseline_handoff.zip`。将它与SHA上传到现有训练目录，校验后解压；只增加包装脚本，不覆盖冻结文件：

```bash
cd /root/autodl-tmp/p1_v4_training
sha256sum -c p1_minimum_baseline_handoff.zip.sha256
unzip p1_minimum_baseline_handoff.zip
nohup bash 07_minimum_baselines.sh > outputs/research_v4_cuda/minimum_baselines.log 2>&1 < /dev/null &
bash 04_status.sh
tail -n 30 outputs/research_v4_cuda/minimum_baselines.log
```

脚本先查recipe完整，再串行R0→H0；各队先实际CUDA检查，按锁恢复，完成后自动打包。单GPU/包装锁避免重入，首个失败停止保留日志；不调整batch、不跳seed、不追加试验。实际新包装的CUDA执行待服务器完成，本地只做语法/Bash检查。

下载`baseline_r0_results.tar.gz`与`.sha256`、`baseline_h0_results.tar.gz`与`.sha256`；核验后再关闭实例。R0网络规模与G0接近，可参考本轮每seed约6.5分钟的epoch循环实测再加I/O/最终汇总；H0另测前5轮，不沿用G0时间承诺。后续本地统一验收、补充主表和部署重载，不重新选测试阈值。

## C. 核心机制：只有一个方案通过前置关口后再登记

先做目标性盲审，重点检查多模型一致漏检的弱点、G0/Spotiflow独有检出及256组问题。新增`09_recipe_analysis_20261102/manual_review_priority.csv`共372个参考点，可先查前20条一致漏检与10条方法分歧；方法计数另存私有key，复核表只链接无预测的原生显示图及原TIF。这个30点样本用于检验瓶颈假说，不估计全数据标注错误率。60图完整匿名审查继续pending，原标签不改，不能自动删除困难真值。

R0/H0结果回来后，锁定一个瓶颈和一个方案：

- 若H0在定位/弱点上有实质收益，可将其作为表示底座，再研究弱候选证据与定位质量的有效耦合。
- 若H0也无收益，优先检验弱候选与背景误检的判别，而非直接增加分辨率或多槽。当前验证0格碰撞，计数上限不是主要解释。
- 先对照相关工作；简单heatmap+offset、flow、注意力、置信度重标定和多分支本身不算已经成立的贡献。方案必须给出超出这些已有设计的具体交互与可检验预测。

最终机制矩阵继续使用base/capacity/full/minus四角色×三seed，但当前**0项启用**。先冻结实现、容量控制、合法移除定义和共同训练配方，再另立有限预算；不将已有空模板视为训练完成。不因本轮无提升就跳过必要参考，也不同时开放多个新机制搜索。

## D. 当前后置事项与研究停止条件

- Spotiflow scratch三次：仅当最终主张需要剥离预训练贡献/从头训练架构比较时启用；当前强外部适配参考已具备，暂不占本次最短补齐预算。既定scratch只是原配方预训练控制，不能称充分调优的外部最优模型。
- 几何/强度拆分、长训练和额外超参搜索：本轮未建立整体收益，暂不追加。若后续科学问题确需拆分，应另立预声明方案，不用于追逐星号。
- 独立新数据：继续按[采集与盲标注SPEC](external_validation_spec_20261102.md)准备；外部读出前锁定最终方法与对比。现有22图已经用于开发，不能重切220来制造独立验证。

最小benchmark完成后，若仍无可支持的新核心机制，交付工程/数据/适配比较版本并明确其研究定位；停止把模块堆叠和更多训练等同于架构论文完成。若核心确有净收益，再补必要消融与独立数据，形成最终研究交付。
