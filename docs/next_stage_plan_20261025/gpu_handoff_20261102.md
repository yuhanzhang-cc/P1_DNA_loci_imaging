# GPU交接：v4配方对照与baseline队列

执行后更新：recipe六次已完成并验收，增强未建立整体收益；下一步使用[最短收尾SPEC](short_term_closure_20261102.md)的最小R0/H0包装任务。下文保留原训练包操作说明，已完成队列不要重复新建训练。

本地准备、包校验及最终状态见[11月2日汇报](overnight_report_20261102.md)。本轮没有启动正式GPU训练；默认先执行六次recipe，不自动串联其余九次baseline。包包含私有训练数据或权重，仅供上传，不提交GitHub。

## 租用条件与上传

- 推荐单卡RTX4090、24GB显存；CPU至少8核、内存至少32GB、空闲磁盘至少30GB。
- Linux Ubuntu22.04、Python3.12、NVIDIA驱动支持CUDA12.8。PyTorch镜像仅作为Python入口，脚本另建`.venv_cuda`，安装锁定的torch2.11.0/cu128与相关依赖。
- 使用`outputs/next_stage_v4_recipe_20261101/04_bundles/latest.json`定位最终包和SHA256。优先只上传`p1_v4_training.zip`及其`.sha256`；`p1_v4_external.zip`供后续新数据评价，不必现在上传。
- 包已在本地独立解压、核对全部manifest、检查Bash语法和模型重载；Linux依赖下载安装与真实CUDA检查尚待服务器执行。不要直接复用旧`p1_spotiflow`目录或旧输出。

将ZIP和SHA文件放到`/root/autodl-tmp`。新开SSH窗口后执行：

```bash
cd /root/autodl-tmp
sha256sum -c p1_v4_training.zip.sha256
unzip p1_v4_training.zip
cd p1_v4_training
bash 01_setup.sh
bash 02_smoke.sh recipe
bash 03_start.sh recipe
bash 04_status.sh
```

`02_smoke`先做独立合成恢复检查，再检查真实256/512图的batch8输入和当前队列适配器。它会执行少量丢弃的工程更新，不计入正式实验。正式运行会检查CUDA、检查记录、manifest和单GPU锁。若OOM、非有限loss或环境错误，保留日志，修正环境或新建协议版本；不静默降batch、换后端或跳过失败种子。

六项任务为`G0_ref_cuda`/`G0_aug_cuda`各seed42/43/44；100epoch、每项2200更新、20个checkpoint×31阈值。联合增强为D4和zscore后强度仿射。初始权重和采样序列成对一致，验证不增强。此轮是训练配方对照，不能当作架构创新。

进程独立于SSH窗口运行。重连后：

```bash
cd /root/autodl-tmp/p1_v4_training
bash 04_status.sh
tail -n 30 outputs/research_v4_cuda/recipe_runner.log
```

从前5个epoch日志估算剩余时间；包括每5轮的全图推理/选择成本。当前没有在新服务器实测，不给出精确完成时间。若进程异常停止，排除原因后：

```bash
bash 06_resume.sh recipe
```

恢复从已提交epoch边界开始；未提交快照保留后重放。源代码、manifest、协议或设备环境改变时拒绝混续；不得为了兼容错误而删除注册记录。

## 完成、下载与本地分析

`05_pack_results`会先校验六项完整及所选权重SHA；未完成时拒绝打包。

```bash
bash 05_pack_results.sh recipe
ls -lh recipe_results.tar.gz recipe_results.tar.gz.sha256
```

下载两文件到本地一个新目录，例如`D:\Projects\P1_DNA_loci_imaging\outputs\next_stage_v4_recipe_20261101\08_returned_gpu\`。下载后核对SHA，再解压和分析；确认归档可读、模型和指标可验收后再关服务器。结果包保留选定权重、逐图/点结果、协议和日志，epoch缓存及恢复点仍在服务器。

```powershell
Get-FileHash .\recipe_results.tar.gz -Algorithm SHA256
tar -xzf .\recipe_results.tar.gz
D:\Projects\P1_DNA_loci_imaging\.venv\spotiflow_cpu\Scripts\python.exe `
  D:\Projects\P1_DNA_loci_imaging\deepblink_pytorch\scripts\analyze_research_v4.py `
  --results-root .\outputs\research_v4_cuda\recipe --output .\analysis_recipe
```

汇总拒绝未完成或只挑最佳seed；输出均值±sample SD柱图、配对线图、`G0_aug_cuda−G0_ref_cuda`精确配对检验及图像条件bootstrap。三种子双侧精确p最小0.25，不能承诺星号。选择仍使用开发验证集，不能报独立泛化成绩。

## 可选baseline：每次显式启动一队

确认recipe结果和预算后，可在同包、同环境依次执行：

```bash
bash 02_smoke.sh baseline_r0
bash 03_start.sh baseline_r0
# 完成后：bash 05_pack_results.sh baseline_r0

bash 02_smoke.sh baseline_h0
bash 03_start.sh baseline_h0
# 完成后：bash 05_pack_results.sh baseline_h0

bash 02_smoke.sh spotiflow_scratch
bash 03_start.sh spotiflow_scratch
# 完成后：bash 05_pack_results.sh spotiflow_scratch
```

上述命令应逐队等待完成，不要一次粘贴并同时启动。每队三种子、100轮；R0/H0每项2200更新，scratch每项4400更新。scratch用同Spotiflow配置随机初始化，原生batch4/crop256/训练损失；这是预训练控制，尚未充分调优。H0为dense heatmap+offset参考。方法间训练策略不同，不能把完整方法差异直接归因于某个结构模块。

## 新数据与核心消融

外部包冻结R0/G0/Spotiflow适配各三份权重与阈值，附带采集表和禁止重用的开发像素/来源清单。采集与盲标注流程见[新数据交接草案](external_validation_spec_20261102.md)。收集真实生物学分组、采集批次和原始像素标注后，在外部包运行：

```bash
bash 01_setup.sh
bash 02_verify.sh
bash 03_evaluate.sh /absolute/path/new_dataset/manifest.csv cuda
.venv_cuda/bin/python deepblink_pytorch/scripts/analyze_research_v4.py \
  --external --results-root outputs/external_evaluation --output outputs/external_analysis
```

禁止在新数据上重新选择阈值；当前支持原始FOV≤512，不自动缩放大图。外部包预声明Spotiflow_F−G0和G0−R0两个对比，按生物学组bootstrap。元数据仍需人工确认独立性。

核心结构base/capacity/full/minus四角色模板位于`06_study_scaffold/core_ablation_template.yaml`。当前全禁用、无新核心模型；可用`validate_ablation_plan.py`检查模板。需要先通过人工错误复核、相关工作差异、容量对照和机制移除设计，再开新版本训练矩阵。当前模板不能启动新架构实验。
