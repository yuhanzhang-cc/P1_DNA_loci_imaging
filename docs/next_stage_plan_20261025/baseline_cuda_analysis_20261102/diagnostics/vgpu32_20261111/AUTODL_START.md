# AutoDL：独立vGPU32包

上传两个文件，均在本地`outputs/next_stage_v4_recipe_20261101/22_vgpu32_handoff_20261111/`：

```text
p1_vgpu32_training_20261111.zip
p1_vgpu32_training_20261111.zip.sha256
```

租一张vGPU-32GB（预期Ada/RTX4080 SUPER、计算能力8.9、约32GB显存），Python3.12/Linux、支持CUDA12.8的NVIDIA驱动；CPU建议至少8核、内存32GB，数据盘至少50GB且安装前留30GB空间，安装后至少12GiB空闲。镜像预装PyTorch版本不决定本实验版本：脚本会在单独venv安装torch2.11.0+cu128及已锁依赖。不要在CPU或其他未知GPU上绕过检查。

在上传目录执行（全新目录，勿覆盖原5090目录）：

```bash
cd /root/autodl-tmp
sha256sum -c p1_vgpu32_training_20261111.zip.sha256
test ! -e p1_vgpu32_training_20261111 || { echo 'Existing directory protected'; exit 1; }
unzip p1_vgpu32_training_20261111.zip
cd p1_vgpu32_training_20261111
bash 00_verify.sh
bash 01_setup.sh
export P1_PYTHON="$PWD/p1_v4_training/.venv_cuda/bin/python"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash 02_import_review.sh
bash 03_plan.sh
bash 08_hardware_preflight.sh
# 仅当上面成功退出后启动：
bash 04_start.sh
bash 05_status.sh
```

若复用旧环境，跳过`01_setup.sh`，把`P1_PYTHON`设为实际已验证的Python绝对路径；仍须通过全部预检。原始研究审批文件已装入并冻结，`02_import_review.sh`只确认原件，无需再填表。

`08_hardware_preflight.sh`不训练：核验软件/硬件、3 seed历史初始化哈希，并对14份冻结参考在22张validation上强制重新推理。TP/FP/FN、每图预测数须一致，匹配坐标最大差异≤0.005px。它不调用优化器，不选新阈值，不读取test。运行记录在`outputs/hardware_transfer/`。检查也被`04_start.sh`强制调用；复用成功记录前重验环境、设备身份和证据哈希。

若GPU名称、计算能力、初始化、推理或恢复身份不符，先停住并保存以下内容；不要改配置或把旧结果搬进来续跑：

```bash
nvidia-smi
bash 05_status.sh
find outputs/hardware_transfer -name doctor.json -o -name failed.json
bash 07_failure.sh
```

把具体失败日志带回本地分析。确认环境未变后的中断恢复使用`bash 06_resume.sh`。跨机器/显卡迁移须新开一轨，不能混续一个seed。

训练会顺序跑原9次：S0/P0六次及scratch三次，随后评价、图件、英文草稿、计时和打包。没有新卡实测耗时；请预留半天到一天及排错余量，首seed完成后以`05_status.sh`更新估计。

```bash
bash 05_status.sh
tail -n 100 outputs/completion/run_*.log
```

完成后下载`outputs/completion/status.json`中`all_results.path`的ZIP及`.sha256`。总包内有原训练包、含硬件预检证据的报告包、冻结模型评价包。本地验收前保留服务器文件：

```powershell
& .\.venv\spotiflow_cpu\Scripts\python.exe deepblink_pytorch/scripts/diagnostics/completion_v1/cli.py ingest-all --archive "<all_results_x.zip>" --checksum "<all_results_x.zip.sha256>" --source-manifest outputs/next_stage_v4_recipe_20261101/22_vgpu32_handoff_20261111/p1_vgpu32_training_20261111/completion_bundle_manifest.json --output "<全新返回目录>"
```

此包含私有开发图像，勿上传GitHub。所有新结果标记`vgpu32_development_v1`。以后有5090资源时，用原`21_approved_handoff_20261111/p1_approved_training_20261111_final.zip`另开服务器目录复跑；两个实验轨道独立归档，勿合并不同硬件的seed。
