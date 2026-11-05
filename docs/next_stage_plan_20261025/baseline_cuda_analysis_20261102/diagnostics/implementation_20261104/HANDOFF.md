# 明早审查与 AutoDL 操作

先读 [过夜报告](OVERNIGHT_REPORT_20261104.md)。本包**尚未通过实际 CUDA 验收**；S0/P0 都没有正式训练。无需现在为了等待盲审空占 GPU。

## 本地包

`outputs/next_stage_v4_recipe_20261101/18_mechanism_ready_20261104/p1_signed_offset_v1r3.zip`

57,800,419 bytes，95 个成员；配套同名 `.zip.sha256` 使用 LF 换行。

SHA256：`cc336e842605ab4e86cf06ce21e736dcb013d39846da57ba3328688d881454d1`

包里含私有开发图像与标注，留在本地/AutoDL；不要上传 GitHub。旧版 v1 是中间构建，上传 **v1r3**。

## 服务器先只验收，不训练

在新目录解压，保留旧 `p1_v4_training`。比较历史 H0 时优先使用其已记录的 CUDA 环境；本轮没有测试新的 GPU/环境组合。

```bash
cd /root/autodl-tmp
sha256sum -c p1_signed_offset_v1r3.zip.sha256
unzip p1_signed_offset_v1r3.zip
cd p1_signed_offset_v1r3
bash 00_verify.sh
export P1_PYTHON=/root/autodl-tmp/p1_v4_training/.venv_cuda/bin/python
# 仅当原环境不存在时，在这个新目录 bash 01_setup.sh 建立独立环境。
bash 04_status.sh
bash 07_plan.sh S0
bash 02_smoke.sh S0
```

`02_smoke` 仅一个独特合成更新、重放一次，CUDA batch8/canvas512；并检查真实开发图像目标编码及无梯度 FOV 前向。若出现 `Historical CUDA initialization differs`，**停在这里**，保留 traceback、`environment_actual.txt`/原 `pip freeze`、`nvidia-smi` 及 GPU 型号；不得绕过后开始正式训练。CPU 本地 hash 不同这件事已在报告披露，CUDA 是否一致尚未知。

成功的探针路径为 `outputs/probes/S0_<时间>/checks.json`。核对 `passed=true`、`exact_recovery=true`、CUDA/full geometry 和历史初始化一致后，将该文件复制为 `decisions/S0_probe.json`。

## 人工决定后才启动 S0 seed42

1. 审查 `adapter.py` 中唯一的 offset 激活变化，确认接受此对照。
2. 复制 `configs/approval.template.json` 为 `decisions/S0.json`，由真实审查者填写 reviewer/date；`mechanism_approved=true`、`synthetic_fixture=false`。配置 SHA 用下面 dry-run 的 `config_sha256`，不能使用模板空值。S0 不强制 P0 的完整盲审参数决定。

```bash
"${P1_PYTHON:-.venv_cuda/bin/python}" deepblink_pytorch/scripts/diagnostics/mechanism_v1/cli.py --config configs/S0.json --dry-run
bash 03_start.sh S0 42
bash 04_status.sh
tail -n 80 outputs/controller/S0_*.log
# 检查 seed42 输出后再决定后续种子；不要重复运行 seed42 代替 resume。
bash 05_pack_results.sh S0 42
```

正常完成标准：100 epoch、2200 updates、20 snapshots、620 epoch-threshold pairs；选定模型、候选和结果哈希一致。下载结果 `.tar.gz` 及 `.sha256`，本地验哈希后再关闭服务器。打包器包含有效源码、两种选定模型、恢复状态、完整 ranked AP/排序表、模型锁和被选中快照的候选缓存，避免将所有历史快照模型重复塞入返回包；不回传原开发图像。正式输出只允许写入本包的 `outputs/trials/`，非空且未登记的目录受保护。该正式收尾路径本轮未运行，AP/模型锁导出与目录保护分别通过了合成单测。

## P0 及剩余种子

- 先完成 60 例真实盲审；空表、合成表、来源映射列都不应作为启动证据。
- 将 `configs/` 复制到新配置修订目录，回填 P0 半径、归属与归一化。实现的归属为 `nearest_truth_then_index`，归一化为 `per_truth_mean`；半径 2 只是一项尚未批准的候选。不要直接修改密封包内的文件。
- 在本地用 `module_cli.py --config-dir <新配置目录> --data-root . --output <全新包目录>` 重建密封包，重新上传并运行 P0 CUDA smoke。
- P0 approval 还须给出真实完整匿名评分 CSV 的相对路径和 SHA256；保留填写原件。运行器检查 60 个预期匿名 ID、必填答案、审查者及非合成来源，不需要把私有解盲映射传到服务器。
- S0/P0 seed42 开发筛查后，按既有提案审视共同 TP 定位、3px F1、重复 FP 和弱点 Recall。暂定继续条件：定位改善至少 0.10px、F1 不下降超过 0.005、近邻重复 FP 不增加；该规则仍待机制审查确认，不自动宣称通过。
- 再批准其余种子时使用 `bash 03_start.sh S0 43 44`、P0 对应命令；中断则 `bash 06_resume.sh S0 43 44`，必须保持配置/代码/GPU/环境不变。

本地无法验证当前 GPU 的实际显存与耗时；用 seed42 实际每轮与快照耗时更新估计。没有必要把未完成的人工决策转换成额外的普通 baseline 训练。
