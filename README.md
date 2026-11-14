# DNA Loci Imaging

无训练骨架（2026-11-03）：S0/P0、等参数对照、消融、独立test/ranked AP及盲审回填接口已分离为 **scaffold only, not run**。入口与证据见 [骨架交接](docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/scaffold_20261103/README.md)。不改现有模型或结论，真实训练计数仍为41；新方法均为not_started，未启用的参数等待盲审/机制确认。

最新结果：[R0/H0 CUDA验收与比较](docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102.md)。六次最小基线训练及H0恢复已验收，总训练记录41次。相同22图开发验证集上，R0/H0/G0参考F1为0.7887/0.8088/0.8197；H0定位误差较大，保留G0为简单候选。三种子差异均未达统计显著。

当前交付：[固定模型推理候选版](docs/next_stage_plan_20261025/detection_release_20261102.md)；后续按[最短收尾SPEC](docs/next_stage_plan_20261025/short_term_closure_20261102.md)完成峰值/offset对齐诊断与盲审，再锁定单一核心机制及容量对照。R0/H0队列已完成，Spotiflow scratch后置。核心架构收益与独立泛化仍待建立。[前次增强分析](docs/next_stage_plan_20261025/recipe_cuda_analysis_20261102.md)保留为历史阶段记录。

原过夜设计：[SPEC与验收标准](docs/next_stage_plan_20261025/overnight_spec_20261101.md)，保留提案快照；本轮实际进度以11月2日汇报和最终验收JSON为准。

## 2026-10-30：Spotiflow CUDA结果已回收

三种子适配、回传和本地验收已完成。同22张开发验证图上，Spotiflow适配F1为**0.8606±0.0034**，G0为**0.8251±0.0038**。Spotiflow更少误检，G0保留弱点Recall和计数误差方面的取舍；自有核心架构领先、充分调优和独立验证尚未成立。

最新入口：[CUDA分析报告与图表](docs/next_stage_plan_20261025/cuda_analysis_20261030.md)、[后续scope](docs/next_stage_plan_20261025/next_scope_after_cuda_20261030.md)、[阶段状态](docs/next_stage_plan_20261025/README.md)。模型和私有审查材料保留本地，仓库整理聚合材料与复核代码。

## 220版训练与模型归档

当前入口：[220版交付说明与文件校验](docs/clean220_release_20261026/README.md)。包含混合256/512图像训练、合并随机划分的三种子模型、同留出集数据扩充对照，以及首轮四项结构筛查的代码、检查点和结果。当前保留原DeepBlink主干，尚未建立结构改动收益；历史、内部测试和验证筛查成绩分开记录。

后续设计与执行状态见[阶段计划](docs/next_stage_plan_20261025/README.md)及[第二轮scope](docs/next_stage_plan_20261025/round2_20261026/README.md)。结构筛查、关键消融和Spotiflow单配置三种子适配已有结果，未来计划与完成实验分开登记。大型图像NPZ与批量叠图保留本地，交付说明提供固定划分重建方法；原始数据不因本次归档而修改。

## 项目仓库结构

### 根目录

- `data/`：存放项目数据，按 `external`、`raw`、`processed` 分层管理
- `src/`：存放项目脚本。当前既有原始实验脚本，也有已整理的标注工作流脚本
- `tests/`：存放示例数据、50 例测试数据和预处理测试结果
- `outputs/`：存放训练产物、预测结果和中间输出
- `environment.yml`：项目环境定义文件
- `AGENTS.md`：仓库协作与自动化agent工作约束

### `data/`

- `data/external/`：外部来源数据或说明文件
- `data/raw/`：标注工作的原始数据，在提交后禁止修改
  - `tif/`：由 Fiji 导出的单张原始标注图，命名为 `sample_N.tif`
  - `png/`：TrackMate 标点后的结果图，命名为 `sample_N.png`
  - `csv/`：由 TrackMate XML 转出的点坐标表，命名为 `sample_N.csv`
  - `logs/trackmate_xml/`：手动从 TrackMate 保存的 XML
  - `logs/annotation_state/`：Fiji 宏写出的 sidecar 元数据，用于后续补全日志
  - `logs/annoation_record.csv`：原始标注总日志
- `data/processed/`：预处理后的数据或后续分析产物，从`raw/`的数据得出

### `src/`

- `src/annotation_workflow/`：手工标注流程对应脚本
  - `fiji_prepare_annotation.ijm`：Fiji 宏。负责从当前图像导出 `sample_N.tif`，打开 TrackMate，并在完成后保存 PNG 与 sidecar
  - `annotation_tools.py`：日志与 XML 收尾工具。负责将 TrackMate XML 转为最终 `sample_N.csv`，并追加 `annoation_record.csv`
  - `remove_latest_sample.py`：回退工具。删除编号最大的 `sample_N` 相关文件
  - `README.md`：标注工作流说明
- `src/padding/`：与图像和标注坐标补零、统一尺寸相关的脚本
- `src/py/`：

### `tests/`

- `tests/raw_test50/`
  50 例原始测试数据，包含 `raw_tif_50`、`raw_png_50`、`raw_csv_50` 和日志
- `tests/processed_test50/`
  对上述 50 例做 padding、坐标替换、加 header 等处理后的结果
- `tests/sample_pics/`
  小规模示例图片，便于快速试验脚本

## `annoation_record.csv` 字段说明

| 列名 | 含义 |
| --- | --- |
| `date` | 本条标注记录写入日期，格式为 `YYYY-MM-DD` |
| `source_path` | 原始源文件路径。对于 `.mvd2`，这里记录 `.mvd2` 容器文件路径 |
| `source_file_name` | 当前实际标注对象名称。普通 tiff/nd2 通常是源文件名；`.mvd2` 为当前 series 名；若做了 Max Intensity 投影，前面会加 `MAX_` |
| `frame` | 从原图中选定的时间帧编号 |
| `z_axis` | 选定的 z 层编号。若做了 Max Intensity 投影，或原图无 z 轴，则记为 `0` |
| `channel` | 实际用于标注的 channel 编号，从 `1` 开始 |
| `raw_tif_name` | 导出的单张 tif 文件名，如 `sample_17.tif` |
| `raw_csv_name` | 与该图对应的点坐标 csv 文件名，如 `sample_17.csv` |
| `raw_png_name` | 与该图对应的标点 png 文件名，如 `sample_17.png` |
| `x_pixel` | 导出 tif 的图像宽度，单位为像素 |
| `y_pixel` | 导出 tif 的图像高度，单位为像素 |
| `micron_per_pixel_x` | x 方向每个像素对应的实际长度，单位为 micron/pixel |
| `micron_per_pixel_y` | y 方向每个像素对应的实际长度，单位为 micron/pixel |
| `estimated_object_diameter` | 最终采用的 TrackMate `estimated object diameter`。当前逻辑以 XML 中最后一次 `DetectorSettings` 的半径换算为直径 |
| `quality_threshold` | 最终采用的 TrackMate `quality threshold`。当前逻辑以 XML 中最后一次 `DetectorSettings` 的阈值为准 |

## 提交规范

### 规则

推荐使用三段式：

```
<type>(<scope>): <subject>
```

其中：

- `type`：必填，说明**提交性质**
- `scope`：可选，说明**影响范围**， 可写模块、目录、数据批次或子系统，例如 `raw`、`training`、`annotation_workflow`、`padding`、`readme`
- `subject`：必填，**简要描述本次提交内容**，可使用中文，控制在 50 字以内，用短句直接说明改动，不加句号
- 一次 commit 尽量只做一类事情

### 推荐 `type`

- `feat`：新增功能、新增流程、新增接口、新增数据或新增可直接使用的能力
- `fix`：修复 bug、修正错误结果、修复脚本、训练、数据处理或配置问题
- `docs`：修改 README、流程说明、注释、规范文档
- `refactor`：重构目录、整理脚本结构、调整代码组织但不改变主要行为
- `perf`：优化性能、资源占用、运行速度、训练效率或使用体验
- `test`：增加或更新测试、验证脚本、基准样例、检查流程
- `chore`：环境、配置、辅助工具、依赖、`.gitignore`、非业务性维护
- `style`：不影响运行结果的格式整理，如排版、命名微调、空格缩进
- `revert`：回滚某次提交或某批修改

### 示例

```text
feat(raw): 添加100例原始数据
feat(training): 增加新一版训练脚本
fix(annotation_workflow): 修复tiff提取frame错误
fix(training): 修正模型推理阈值逻辑
docs(readme): 补充项目结构与提交规范
perf(padding): 优化大图补零处理速度
chore(env): 补齐环境依赖定义
```

### 适用建议

- 新功能、新数据、新流程：`feat(...)`
- 修 bug 或修结果错误：`fix(...)`
- 只改文档：`docs(...)`
- 只改结构不改行为：`refactor(...)`
- 做性能或效率优化：`perf(...)`
- 改环境、依赖、配置、辅助脚本：`chore(...)`
- 回滚错误提交：`revert(...)`

Latest no-training implementation and GPU handoff (2026-11-04): [S0/P0 overnight report](docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/implementation_20261104/OVERNIGHT_REPORT_20261104.md).

Latest review-to-report preparation (2026-11-11): [overnight report](docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/handoff_20261111/OVERNIGHT_PREFAB_REPORT.md) and [morning steps](docs/next_stage_plan_20261025/baseline_cuda_analysis_20261102/diagnostics/handoff_20261111/START_HERE.md).
