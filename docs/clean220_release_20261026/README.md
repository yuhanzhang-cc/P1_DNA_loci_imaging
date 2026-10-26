# clean220训练、模型与证据归档

归档日期：2026-10-26。保存220版迭代至首轮结构筛查结束的工作，以及下一轮设计交接。没有启动v3训练，也没有将负结果标作新方法收益。

## 模型、协议与成绩入口

| 目录 | 模型文件 | 状态与解释 |
|---|---|---|
| [261024_clean220_random](../../outputs/261024_clean220_random/最终汇报.md) | `seed_42/43/44/best_model.pt`，`selected_model.pt`，共4份 | 当前合并随机176/22/22的220版；内部测试三种子F1=0.7716±0.0327，选定seed42为0.8071 |
| [261025_clean220_control](../../outputs/261025_clean220_control/最终汇报.md) | 三种子`best_model.pt`及`selected_control_model.pt`，共4份 | 旧批次89训练图与扩充176图使用同留出集；内部均值差+12.77个百分点，属于数据扩充对照，不是架构收益 |
| [next_stage_v2](../../outputs/next_stage_v2/阶段汇报.md) | A/B/D/D2各一份`best_model.pt`，共4份 | train/valid-only的结构筛查：参考验证F1=0.828229，A0.815303、B0.789986、D0.807588、D2 0.794798；没有结构超过参考 |
| [261019_clean220](../../outputs/261019_clean220/最终汇报.md) | 三种子、推荐模型、历史模型固定画布派生，共5份 | 早期分源划分的220实验，保留溯源；不作为当前合并随机协议的直接成绩 |

合计17份检查点；派生/推荐文件不代表17个独立训练模型。另保存7份v2验证预测网格，内容仅为`grids/sample_id/original_shape`，用于复核既有验证阈值表，不含原始图像。权重保持原始字节，未删除optimizer state或重新序列化。每份checkpoint均附原实验配置/日志，实际文件和SHA256见[artifact_manifest.json](artifact_manifest.json)。

原113版0.8488属于自己的历史测试及阈值选择，不能与上述分数直接相减判断模型进退。v2验证分数和前一轮内部测试分数也不能混为一列。当前22张内部测试图已经观察过；独立新图像验证与人工盲法标注复核仍未完成。

## 提交内容与本地保留项

归档包含：固定划分、数据审计、三种子训练日志/曲线、检查点、验证/测试逐图指标与点匹配、版本比较、数据对照、首轮四项结构及其失败结果、运行环境记录、训练时源码快照和最终交付源码快照。代码包含混合尺寸预处理、padding损失mask、XPU选项、development-only读取和点匹配/边界得分修复。

本地保留而不上传：原始图像NPZ及按图像生成的大批叠图、局部图像patch联系表、缓存、运行PID和环境目录；无关申请书不属于此包。文件不删除。省略项及大小也记录于manifest。图像归档按固定划分从已有原始输入重建；不把整个`outputs`目录打成一个不可审阅的压缩包。

`.gitignore`仅对明确列出的检查点/预测网格放行，其余NPZ/PT仍默认忽略。`.gitattributes`对本轮冻结产物和计划禁用换行转换，以保护已记录的字节哈希；没有改写旧报告或源码快照。

## 校验与使用

在仓库根目录运行，只使用Python标准库，不加载图像或执行训练：

```text
python docs/clean220_release_20261026/verify_archive.py
```

该命令逐文件校验manifest中已发布产物的大小和SHA256；省略的本地文件不要求存在。实际发布前验证记录见[validation.json](validation.json)。模型为本项目本地训练产物；校验脚本不会反序列化checkpoint。

推理需要Python≥3.12及`deepblink_pytorch/pyproject.toml`依赖。共享环境定义不变；可选[requirements-xpu.txt](../../deepblink_pytorch/requirements-xpu.txt)固定Intel XPU torch，仅用于隔离环境，不替换团队的共享环境。以下示例用于**新图像目录**：

```text
python -m pip install -e ./deepblink_pytorch
deepblink-torch predict --model outputs/261024_clean220_random/selected_model.pt --images NEW_IMAGE_DIR --output NEW_PREDICTION_DIR --probability 0.55 --device cpu --output-unit px
```

`0.55`为该交付模型原协议的验证阈值；v2对同一seed42主干权重的验证阈值为0.12678571428571428，使用不同协议时需明确写出。CLI默认0.5不会自动读取校准阈值。输出单位默认原像素，不能把缺少标定的默认1.0解释为真实1µm/px。

## 重建图像归档与历史复现边界

固定清单为[pooled frozen_split.csv](../../outputs/261024_clean220_random/frozen_split.csv)。已有原始输入时，可在新目录重建，不覆盖冻结证据：

```text
deepblink-torch create --images data/raw/tif --labels data/raw/csv --output outputs/local_rebuild/dna_clean220_random.npz --input-size 512 --exclude-samples sample_230,sample_286 --split-manifest outputs/261024_clean220_random/frozen_split.csv --seed 20261024
```

处理为原图有效像素归一化，再向右/下补零512；坐标不缩放，人工padding占据标为−1。该命令只生成数据，不启动训练；新生成ZIP/NPZ的容器字节不保证与历史文件相同，应核对数组、原尺寸、样本顺序及manifest。

精确历史重跑应在独立checkout恢复相应实验的训练时源码快照及配置，并使用新的输出目录。旧control runner会主动拒绝与扩充组不一致的核心代码hash；当前代码已经包含后续修复，不能删除校验后把重跑称为原始历史试验。不要在冻结产物目录重跑会改写审计文件的旧prepare/summary命令。训练脚本提供入口，快照提供实际训练版本；硬件和数值差异也需记录。

## 下一分支的起点

[总计划](../next_stage_plan_20261025/README.md)和[第二轮scope](../next_stage_plan_20261025/round2_20261026/README.md)包含R1细节直达、R2上下文范围选择、H0密集输出参考、普通替代控制及最多12次新训练的草案。旧52次全研究估算的剩余阶段挂起。第二轮代码尚未实现，不能将本包当作v3完成版本。

VS Code审阅重点：训练/推理几何与padding一致性；点匹配及offset边界得分；协议与成绩版本；17份权重的明确放行；下一轮预算和机制控制。没有新增共享环境依赖或运行新的训练。
