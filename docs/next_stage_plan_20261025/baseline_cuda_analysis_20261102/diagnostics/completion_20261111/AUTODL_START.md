# AutoDL 启动：使用已审批的最终包

本地材料齐全：60/60真实审查、P0半径2、完整S0/P0六次、Spotiflow scratch三次均已批准并冻结。原件已移入 `outputs/next_stage_v4_recipe_20261101/21_approved_handoff_20261111/received/`，SHA256未变。

只上传以下两个文件（其他r1/r2/r3为保留的工程历史，不使用）：

```text
outputs/next_stage_v4_recipe_20261101/21_approved_handoff_20261111/p1_approved_training_20261111_final.zip
outputs/next_stage_v4_recipe_20261101/21_approved_handoff_20261111/p1_approved_training_20261111_final.zip.sha256
```

包内已有两份原件、冻结配置、现有14份模型/确定性参考和开发数据，没有独立test数组或解盲映射。私有ZIP不要上传GitHub。最终字节数和哈希见 `RELEASE.json`。

运行条件沿用现有 matched CUDA 参考：RTX5090、Python3.12、torch2.11.0+cu128；实际初始化哈希必须通过。环境安装后至少12GiB空闲盘；按原交接至少30GB空闲盘预留安装/缓存。没有Linux新服务器实测或新GPU训练成绩，CPU通过不代表CUDA通过。

在服务器上传目录执行：

```bash
cd /root/autodl-tmp
sha256sum -c p1_approved_training_20261111_final.zip.sha256
test ! -e p1_approved_training_20261111_final || { echo 'Existing directory protected; inspect before resuming'; exit 1; }
unzip p1_approved_training_20261111_final.zip
cd p1_approved_training_20261111_final
bash 00_verify.sh

# 如原已验证的环境仍在，复用它：
export P1_PYTHON=/root/autodl-tmp/p1_v4_training/.venv_cuda/bin/python
# 若上面路径不存在，改用本包的全新环境：
# bash 01_setup.sh
# export P1_PYTHON="$PWD/p1_v4_training/.venv_cuda/bin/python"

bash 02_import_review.sh
bash 03_plan.sh
bash 04_start.sh
bash 05_status.sh
```

`02_import_review.sh`只确认已有冻结决定，不要求重新审批。单进程在一张GPU上依次完成：CUDA/初始化预检、S0/P0探针、六次机制训练、三次scratch控制、统一评价、图件、英文草稿、推理计时及总结果打包。即使seed42筛查为负，仍按已批准的full_registered_six完成六次；不新增机制、不改配方、不重跑R0/H0/G0。

进度与日志：

```bash
bash 05_status.sh
tail -n 100 outputs/completion/run_*.log
```

九个任务逐个列出epoch和更新数。ETA只外推epoch回调，不含安装、选模、后处理和下载。优先为白天预留5–8小时及排错余量，首个seed42完成后以实测修正；不是完成时限保证。

若预检失败，先保留日志，不豁免初始化差异：

```bash
bash 07_failure.sh
# 调查后保持同源、同配置、同后端：
bash 06_resume.sh
```

全部完成时，`outputs/completion/status.json` 的 `all_results.path` 是应下载的总ZIP，同时下载其 `.sha256`。它包含原始训练结果包、图文报告包和冻结模型评价包。下载并本地验收前保留服务器文件。

本地一条命令验收：

```powershell
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py ingest-all --archive "<all_results_x.zip>" --checksum "<all_results_x.zip.sha256>" --source-manifest outputs/next_stage_v4_recipe_20261101/21_approved_handoff_20261111/p1_approved_training_20261111_final/completion_bundle_manifest.json --output "<全新返回目录>"
```

独立数据、独立泛化结论和最终期刊排版不在这九次训练中凭空产生。入口已备好，来源审查/真实数据和论文作者信息仍按[完整手册](ONE_DAY_RUNBOOK.md)补齐。
