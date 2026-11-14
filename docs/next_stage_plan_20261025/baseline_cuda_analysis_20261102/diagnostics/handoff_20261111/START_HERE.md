# 明早从这里开始

**你负责图像判断和一次研究决定；配置哈希、探针路径、条件队列、返回包及报告由脚本接力。** 本轮没有替你填写审查结论，也没有开始正式训练。

## 1. 在图上审阅（预计人工 40–60 分钟）

本机打开 [新版匿名审阅页](private/reviewer_v2_final/index.html)。私有图像不进入 GitHub，远程阅读仓库时不会提供这个本机页面。

页面支持直接填答、草稿 JSON 下载/恢复、可选点击建议中心。每例判断圆圈附近是否有可信 spot、整个 patch 的可辨认点数或“不确定”、置信度；填一个真实审查者。建议先看两例确认显示和控件正常，再完成 60 例。真实浏览器点选因工具授权超时未验收，页面逻辑和导出回读已测。

完成后下载 `blind_audit_responses.csv`，保留原件及草稿。**不需要手填坐标，不要求人眼判断 0.23px，也不把本页点击直接变成训练标签。** 不确定项可另用原始 TIFF/Fiji；预计额外 15–30 分钟，取决于数量。以上是估计，不是实测工时。

## 2. 只确认一次研究方案（预计 5–10 分钟）

评分后打开页面中的 [研究决定表](private/reviewer_v2_final/decision.html)，填写确认者，并确认：

- 是否需要先修订标签。若需要，转下面的纠标分支，当前训练不会启动。
- 是否接受已准备的 S0/P0 对照。P0 主提案为半径 **2**，几何依据见 [过夜报告](OVERNIGHT_PREFAB_REPORT.md)；不是要求你凭肉眼估最优半径。若接受，在表中明确选择 2；证据不足就不要批准。
- 继续条件：共同 TP 至少 20，P0 平均定位改善至少 0.10px，F1 不下降超过 0.005，近真值 FP 几何代理不增加。是否允许通过后自动完成剩余四次，由你一次选择。

下载 `research_decision.json`。不需要手写 JSON、SHA256、Python 或批准文件。

## 3. 上传预制包，之后只补两份输入

最新版：`outputs/next_stage_v4_recipe_20261101/19_prefabrication_20261111/p1_review_pipeline_20261111.zip`，约 58 MB，119 个成员。

SHA256：`dc93bf16d730e4cdcac0ab9ce389a774c6435182fb3f72998112d6e1761c3fc2`。同目录 `.zip.sha256` 使用 LF。包内有私有开发数据，**不要提交 GitHub**。旧 v2、v1r3 保留作历史材料，本次用 **20261111 release**，不要覆盖旧云端目录。

已有对照登记为 RTX 5090 / torch 2.11.0+cu128。复用原验证过的环境，或在新目录建立独立环境；实际 CUDA 验收仍未完成。

```bash
cd /root/autodl-tmp
sha256sum -c p1_review_pipeline_20261111.zip.sha256
unzip p1_review_pipeline_20261111.zip
cd p1_review_pipeline_20261111
bash 00_verify.sh
export P1_PYTHON=/root/autodl-tmp/p1_v4_training/.venv_cuda/bin/python
# 若原环境不存在：在本新目录 bash 01_setup.sh；不覆盖已有 venv。
bash 03_plan.sh
bash 05_status.sh
```

将明天导出的两份文件放入这个新包的 `incoming/`，无需重新打包 ZIP：

```bash
bash 02_import_review.sh incoming/blind_audit_responses.csv incoming/research_decision.json
bash 04_start.sh
bash 05_status.sh
tail -n 80 outputs/pipeline/run_*.log
```

一次启动执行：源文件校验 → 真实回填/方案冻结 → CUDA/初始化 doctor → S0/P0 单步探针 → 两个 seed42 → 继续条件 → 通过且已授权才展开四次 → 完成验收 → 自动返回包。

doctor 不通过不会训练；科学继续条件不通过会保存两次筛查结果、自动打包并停止扩展，不会自动换模块或加跑普通 baseline。完整六次的耗时要用实际 seed42 测量后估计，今晚没有 GPU 实测承诺。

中断或失败时：

```bash
bash 08_collect_failure.sh
# 保留旧文件，调查后在同源/同配置/同后端恢复：
bash 06_resume.sh
# 若要先取回已完成的部分：
bash 07_pack_completed.sh
```

成功返回包命名为 `return_manifest_<ID>.tar.gz`，同时产生 `.sha256`。下载 **两者**，本地验收前不要只凭终端 complete 就删除服务器材料。

## 4. 本地一行验收，再一行生成报告

在项目根目录，用现有 CPU 环境运行；以下尖括号是你下载的实际文件名和全新输出目录。分析目录不要放在解压原始结果内部。

```powershell
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py ingest --archive "<下载的tar.gz>" --checksum "<对应sha256>" --source-manifest outputs/next_stage_v4_recipe_20261101/19_prefabrication_20261111/p1_review_pipeline_20261111/bundle_manifest.json --output "<全新解压目录>"
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py report --data-root . --returned "<上步解压目录>" --output "<全新报告目录>"
```

自动产出：每种子 P/R/F1/完整 ranked AP/弱点 Recall/定位误差，均值±SD，计划对比与未完成项，共同 TP 定位和近真值 FP 代理对照、来源记录及 `REPORT.md`。未跑的实验不写零分，不因增加数据或换统计口径直接认定进步。

## 如果确实要修标注

**先完成并锁定盲审，再打开** `private/owner_correction_materials/`：19 张原生图与原点、60 个候选编号已准备。在 Fiji 用 Plugins → Macros → Run 打开相应 `macros/sample_N.ijm`，选该材料目录；原标注为可编辑绿点，候选为青色叠加。也可继续用你保存的原 TrackMate 工程。操作在图上完成，另存完整的该图点列表，不覆盖原标注。这里没有声称已做 TrackMate XML 格式转换。

Fiji 核心的亚像素/叠加/CSV I/O 已用合成数据实测，真实 GUI 拖点仍需首例确认。宏导出只写新修订目录的 `sample_N_reviewed.csv`。本机验证命令见 [README](README.md)。

```powershell
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py check-revision --data-root . --reviewed-csv "<图上编辑导出的CSV>" --coordinate-units pixels --output "<新核验json>"
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py propose-labels --data-root . --reviewed-csv "<CSV1>" "<CSV2>" --reviewer "<实际审查者>" --reason "<纠标依据>" --output "<新修订提案目录>"
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/handoff_v2/cli.py recheck-labels --data-root . --revision "<上步目录>" --output "<冻结模型诊断复评目录>"
```

这些工具保留父数据、图像、split 和旧结果；修订数据只是提案，旧模型用原权重/原阈值诊断复评。**当前正式训练会拒绝把它当成旧数据偷偷续跑。** 新标签版本是否成为正式基准、是否需要重新比较，仍需研究决定，不会自动修改旧结论。
