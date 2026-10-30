# AutoDL：Spotiflow三种子训练操作说明

日期：2026-10-30。数据包仅含176张训练和22张验证图及像素坐标；不含测试集、账号或密钥。
本地已验证：数据/坐标一致、原生默认/校准推理、CPU训练、真实epoch快照、精确中断恢复。
**CUDA和Linux安装尚未在本地实机验证，服务器上必须先通过GPU smoke再开始正式训练。**

## 租用条件

| 项目 | 建议 |
|---|---|
| GPU | 1张NVIDIA RTX 4090 24GB；3090 24GB也可先做预检；不需要多卡 |
| 显存 | 建议24GB，作为稳妥预留，尚非实测最低需求 |
| CPU / RAM | 建议至少8 vCPU / 32GB RAM；运行器固定使用4个Torch线程、num_workers=0 |
| 磁盘 | 建议数据盘100GB可用，容纳独立环境、20快照×3种子、恢复状态和日志 |
| 系统 / Python | Linux x86_64，优先Ubuntu22.04/24.04，**Python3.12**镜像 |
| NVIDIA驱动 | 与CUDA12.8 runtime兼容；先查看`nvidia-smi`，确认镜像能识别GPU |
| PyTorch | 脚本在独立venv安装torch2.11.0 + torchvision0.26.0 CUDA12.8版本，不依赖镜像自带版本 |

这是建议起点，不是租赁动作或速度保证。三种子CPU实测估算12–15小时；GPU耗时要在该实例上测，不能按卡名直接承诺。
只租一张卡、三个种子串行。不要用CPU预检权重充当CUDA的一个正式种子。
CUDA是配套运行环境，不是额外租用的一台设备；无需先手动重装驱动。

## 一、上传和解压

将`spotiflow_dev_cuda_bundle.zip`上传到实例数据盘（通常`/root/autodl-tmp`）。
数据包在本地`outputs/next_stage_spotiflow_20261030/05_cuda_package/`下；SHA256见同目录`SHA256SUMS.txt`。
只需上传zip。GitHub仓库不保存这个含开发图像的数据包。

打开AutoDL JupyterLab终端，执行：

```bash
cd /root/autodl-tmp
python -m zipfile -e spotiflow_dev_cuda_bundle.zip p1_spotiflow
cd p1_spotiflow
```

如果上传目录不同，只改第一行路径。使用新的`p1_spotiflow`文件夹；不要覆盖正在训练的旧任务目录。

## 二、安装独立环境

```bash
nvidia-smi
python --version
bash 01_setup.sh
```

必须为Python3.12。安装脚本创建`.venv_cuda`，安装固定Torch/torchvision、Spotiflow0.6.5、Lightning2.6.6和项目包，执行依赖检查及完整文件校验。
镜像自带PyTorch不必改动；本任务全部命令使用`.venv_cuda/bin/python`。
若下载失败，保留日志后重试安装，不要临时升级Torch或Spotiflow版本。
安装后的实际依赖清单保存在`installed_cuda_environment.txt`。

## 三、GPU小规模预检

```bash
bash 02_smoke.sh
```

预检只用合成数据，核对真实epoch快照、权重重载及连续/中断恢复的模型、优化器、调度器状态。
通过文件为`outputs/spotiflow_cuda/engineering_smoke/checks.json`，须记录`device=cuda`和当前Torch版本。
CPU预检通过不能替代这个文件。预检未通过时，正式入口会拒绝训练。
失败日志见`outputs/spotiflow_cuda/smoke.log`；不要删除失败证据或反复开正式任务。

## 四、后台启动三种子

```bash
bash 03_start.sh
```

这个命令会后台顺序运行seed42、43、44，SSH/Jupyter终端断开不影响任务。
脚本记录PID并拒绝明显重复启动；运行器另有任务锁。不要反复执行。
每次100轮、batch4、实际crop256、44更新/轮；三个种子共13,200次更新。
配置使用general预训练、原生loss/增强，AdamW lr0.0003、wd0.01；不是与本地batch8完全相同算力的实验。
每5轮保存一份**当轮模型**，共20份；在22张完整验证图上统一3px匹配、31阈值选模。
另保留同20快照中的原生loss次级选择。不根据测试集或单个种子成绩改参数。

## 五、查看进度

```bash
bash 04_status.sh
```

持续看训练日志：

```bash
tail -f outputs/spotiflow_cuda/training.log
```

`Ctrl+C`只退出`tail`，不停止后台训练。
关键文件：

```text
outputs/spotiflow_cuda/
  protocol.json
  installed environment: ../../installed_cuda_environment.txt
  training.log
  engineering_smoke/checks.json
  seed_42/status.json
  seed_42/progress.json
  seed_42/training_metrics.csv
  seed_42/recovery/last.ckpt
  seed_42/epoch_005/... through epoch_100/...
  seed_42/selection.json
  seed_42/primary/
  seed_42/native_secondary/
  seed_43/...
  seed_44/...
```

`progress.json`每轮更新；`status=complete`表示该种子训练与最终选择均完成。
首个5轮快照完成后，可用实际epoch/全图评价耗时估计剩余时间；当前没有经过实测的GPU完成时间。

## 六、意外中断后续训

先确认旧进程已结束，并检查日志中的异常原因。如果是源代码、数据、非有限数值或显存错误，先修复/核对，不直接重试。
基础设施中断且配置、数据、环境不变、有完整epoch恢复文件时：

```bash
bash 06_resume.sh
```

它读取`recovery/last.ckpt`，跳过已完整完成的种子；不覆盖既有epoch快照或偷偷从头再训练。
如果最后一次异常发生在快照已写出但该轮完整恢复点尚未落盘的极短窗口，先保留现场并回报，不能直接删除快照解决冲突。
不可把原CPU checkpoint直接当作该CUDA实验的无缝同条件续训。

## 七、结果打包和下载

三种子全部完成后：

```bash
bash 05_pack_results.sh
```

得到当前目录的`spotiflow_cuda_results.tar.gz`，包含模型、20份逐epoch候选与阈值表、完整恢复状态、训练日志和环境清单。
下载这个文件回本地进行统一报告和归档。只下载最后的`best.pt`不足以复核完整选模过程。
下载确认后手动关闭AutoDL实例；脚本不会自动操作账户或关闭付费资源。

## 验证边界与数据使用

- 只有198张开发图，无测试数组；不得临时导入测试图挑阈值或改架构。
- general预训练来源与本项目的潜在同源尚未完全排除，结果不是独立泛化证明。
- 三个种子是训练重复，不是生物学样本数；不以新增显著性星号为运行目标。
- 数据包仅供私有任务使用，不要上传至公开GitHub。

官方说明（2026-10-30核对）：
[PyTorch2.11安装](https://pytorch.org/get-started/previous-versions/#v2110)、
[Spotiflow安装](https://weigertlab.org/spotiflow/installation.html)、
[AutoDL快速开始](https://www.autodl.com/docs/quick_start/)、
[AutoDL数据盘](https://www.autodl.com/docs/local_disk/)。
