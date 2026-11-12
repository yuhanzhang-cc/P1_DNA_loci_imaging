# 审批后一个白天：最短执行路径

**更新：真实60例审查和九次训练审批已接收。现在启动只用[AUTODL_START.md](AUTODL_START.md)中的最终已审批包；下文旧r2文件名仅保留为审批前设计历史，不再上传。**

目标是完成**已定义的开发研究矩阵**及可复核交付，不预定“本方法最好”。没有独立数据、有效机制收益或采集元信息时，不能保证一天内完成一篇可投稿论文。本文是最新命令入口；不要混用旧上传包、旧 `04_status.sh` 和新决定文件。

## 早晨：约 60–100 分钟人工窗口

1. 打开 `private/reviewer_release_r2/index.html`，先确认两例显示正常，再完成 60 例。无需判断0.23px或看CSV猜点；不确定可选uncertain。下载评分CSV，保留草稿备份。
2. 在旁边的 `decision.html` 确认 P0 参数。半径2是已有几何证据支持的提案，不是已证明最佳值。
3. 如果目标是一次完成三种子消融，明确勾选“无论seed42筛查结果如何，完成预先登记的六次试验”。这是**新审批选项**，默认关闭；它改变继续策略，不改训练配方。原“过筛才扩展”仍可选择。
4. 若要补齐外部预训练控制，再勾选 Spotiflow scratch 三种子。它不是充分调参后的最佳scratch方法，不额外重跑R0/H0/G0。
5. 导出新版 `p1_research_decision_v3` JSON。旧上传包会拒绝这个新schema，避免静默忽略新选项。若发现必须纠标，暂停该数据版本训练，使用前一轮19图Fiji材料及版本提案工具；这条科学分支无法承诺固定工时。

## 上传与执行：新目录、一个进程、单GPU

最终包、字节数、SHA256以 `RELEASE.json` 为准。文件位于 `outputs/next_stage_v4_recipe_20261101/20_completion_20261111/`，私有数据包不要提交GitHub。

```bash
cd /root/autodl-tmp
sha256sum -c p1_completion_20261111_r2.zip.sha256
unzip p1_completion_20261111_r2.zip
cd p1_completion_20261111_r2
bash 00_verify.sh
bash 03_plan.sh
# 复用你此前保留且已验证的环境；路径不存在才用本包01_setup创建新环境。
export P1_PYTHON=/root/autodl-tmp/p1_v4_training/.venv_cuda/bin/python
# 若没有可复用环境：bash 01_setup.sh
# export P1_PYTHON="$PWD/p1_v4_training/.venv_cuda/bin/python"
bash 02_import_review.sh incoming/blind_audit_responses.csv incoming/research_decision.json
bash 04_start.sh
bash 05_status.sh
tail -n 100 outputs/completion/run_*.log
```

现有 matched CUDA 锁登记为 **RTX5090 / torch2.11.0+cu128**。这是历史协议条件，不是新硬件性价比推荐。实际CUDA、三种子初始化哈希与源代码绑定未通过就停止取证，不擅自豁免。首次安装网络依赖尚未在新服务器实测；版本定义沿用包内已锁定requirements，不升级共享本地环境。

环境安装后至少保留12GiB空闲数据盘，运行器会在训练前检查；准备实例时按既有至少30GB空闲盘的交接条件预留安装/缓存空间。不要通过减batch、换AMP或删除未下载结果来绕过检查。

一键执行顺序：冻结决定 → CUDA doctor → 两个单步探针 → S0/P0 seed42 → 按已批准策略继续余下四次 → 已批准时跑scratch三次 → 验收 → 统一指标/AP/共同TP误差 → 五张开发图及英文草稿 → 同设备冷加载推理计时 → 冻结评价包 → 一个总返回ZIP。

## 预算与进度

历史回调耗时：H0每100轮约260秒；R0/G0约382–391秒；Spotiflow transfer每100轮约340秒（其原硬件/回调定义不同）。这些只支持量级估计，**不含安装、数据准备、最终选模、I/O和新机制性能变化**，不能直接作为新训练耗时保证。

建议预留：环境/上传30–60分钟，最多九次训练及预检1–3小时，分析/计时/打包30–60分钟，下载与复核30–60分钟，再留1–2小时排错。若CUDA首次实测接近历史量级，约5–8小时白天窗口可覆盖已定义开发任务；网络/初始化失败、纠标或新机制决策可能超出。首个seed42完成后用实际墙钟更新ETA。当前不启动付费GPU。

运行状态：`outputs/completion/status.json`；机制细节：`mechanism/outputs/pipeline/status.json`；scratch逐epoch：`p1_v4_training/outputs/research_v4_cuda/spotiflow_scratch/`。最终 `all_results.path` 是需下载的总ZIP，同时下载同名`.sha256`。保留服务器，直到本地完整验收。

中断时：

```bash
bash 07_failure.sh
# 调查后保持同源/同配置/同后端，保留旧目录：
bash 06_resume.sh
```

## 下载后：一条命令验收所有返回组件

从正常项目根目录运行，尖括号替换为实际路径；源manifest保留在本次上传包的本地未压缩目录里。

```powershell
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py ingest-all --archive "<all_results_x.zip>" --checksum "<all_results_x.zip.sha256>" --source-manifest outputs/next_stage_v4_recipe_20261101/20_completion_20261111/p1_completion_20261111_r2/completion_bundle_manifest.json --output "<全新本地返回目录>"
```

`verified_delivery/` 内已有analysis、figures、manuscript和benchmark；`verified_mechanism/`及可选`verified_scratch/`保留可复评模型和选定缓存；`frozen_evaluation.zip`用于后续独立读出。方法未跑、没有改善或只有一个种子时，表和文章不会填零、生成虚构SD或写“我们最好”。

## 如果独立数据已经存在

只需给出真实TIFF、完整`y,x`标签CSV和实际生物学/采集组元信息；模板是本目录 `independent_images.template.csv`。不要把看过的历史内部测试集称为独立。待真实来源审查完成，脚本可自动计算文件哈希，不必手填。

```powershell
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py prepare-independent --metadata-csv "<数据目录/metadata.csv>" --evidence "<数据目录/independence_review.md>" --reviewer "<真实确认者>" --reviewed-at "<实际日期>" --notes "<独立性依据及限制>" --confirm-independence --output "<同数据目录/manifest_v1.json>"
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py independent --root . --analysis "<verified_delivery/analysis>" --manifest "<manifest_v1.json>" --output "<全新独立评价目录>"
# 上一行为只检查元信息和模型锁。确认输入后才增加 --execute 执行冻结模型读出：
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py independent --root . --analysis "<verified_delivery/analysis>" --manifest "<manifest_v1.json>" --output "<全新独立评价目录>" --execute
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py independent-finish --evaluation "<独立评价目录>" --analysis "<verified_delivery/analysis>" --output "<全新独立成文目录>"
```

注意云端分析目录里的catalog有云端绝对路径：本地重做推理前，用 `finish --returned <verified_mechanism> --scratch <verified_scratch>` 生成本地绑定的analysis，或解压 `frozen_evaluation.zip` 并使用其 `--root <包目录> --analysis <包目录>`。前者会复核原数据与模型，后者已带所有冻结权重和排除清单，可直接在有环境的服务器读出。无独立数据时整段留待执行，不以合成测试冒充验证。

## 投稿前仍需作者确认

`author_metadata.template.json` 集中收集真实采集/注释信息、来源重叠解释、图像示例、研究解读及目标期刊。填好并实际review后，`manuscript --author-metadata ...` 或 `independent-finish --author-metadata ...` 可自动回填，不能覆盖计算出的指标。期刊规范需选定期刊后再核验；当前SVG/PDF/300dpi PNG是通用候选输出，不宣称已满足任何期刊。
