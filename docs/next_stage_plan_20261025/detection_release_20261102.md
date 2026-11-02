# 固定模型检测/定位候选版

版本：`p1_loci_detection_rc_20261102`。ZIP及SHA位于本地`outputs/next_stage_v4_recipe_20261101/10_detection_release_20261102/`，约55.6MiB。包含46个文件、两份既有选定权重、依赖定义、源码、模型锁和使用说明；不含原始图、标签或NPZ。

| 入口 | 固定模型 | 开发验证成绩与角色 |
|---|---|---|
| 默认spotiflow | general适配，seed42/epoch60/p=0.6553990357712657 | 同22图micro-F1=0.8600；外部方法的本地适配 |
| 可选g0 | 旧XPU训练G0，seed42/epoch65/p=0.01098694263059318 | 同22图micro-F1=0.8261；既有自有简单参考 |

当前软件候选版用于可复现推理。新CUDA增强实验单独归档，未据其改默认配方；核心架构领先和独立泛化未确立。选择seed42沿用预定代表，不挑本轮最高分种子，也不重新调阈值。

## 本地使用

可直接使用已经安装好的项目Spotiflow环境；脚本从候选版自身加载代码与权重。PowerShell示例：

```powershell
$releaseRoot = "D:\Projects\P1_DNA_loci_imaging\outputs\next_stage_v4_recipe_20261101\10_detection_release_20261102\p1_loci_detection_rc_20261102"
$pythonExe = "D:\Projects\P1_DNA_loci_imaging\.venv\spotiflow_cpu\Scripts\python.exe"
& $pythonExe "$releaseRoot\deepblink_pytorch\scripts\predict_loci_release.py" --verify-only
& $pythonExe "$releaseRoot\deepblink_pytorch\scripts\predict_loci_release.py" --input "D:\YOUR_TIFS" --output "D:\NEW_RESULT_DIRECTORY" --device cpu
```

输入为一个TIFF或一个TIFF文件夹，单通道二维、原生每边≤512。通道/Z选择应显式在上游完成；不自动投影或缩放。Spotiflow沿用native auto normalizer与subpixel/NMS适配，G0沿用有效域zscore后右下补512，坐标始终在原始视野。

输出包括`points/图名.csv`（y,x,p）、`summary.csv`、`run_manifest.json`和`completion.json`。单位为pixel，不猜测微米标定；输入文件和模型的SHA、实际方法、设备、阈值均记录。每次指定新输出目录，保留已有结果。

## 服务器使用

解压候选版后，可复用现有训练环境，指定脚本完整路径；例如：

```bash
/root/autodl-tmp/p1_v4_training/.venv_cuda/bin/python \
  /root/autodl-tmp/p1_loci_detection_rc_20261102/deepblink_pytorch/scripts/predict_loci_release.py \
  --input /path/to/tifs --output /path/to/new_results --device cuda
```

两个入口均已在本地CPU上处理一张真实开发验证图，输出像素范围与状态通过；这不是新增测试集评价或公平速度benchmark，耗时含加载/首次调用。服务器上的新包装CUDA推理仍需实际检查。

长期研究下一步见[最短收尾SPEC](short_term_closure_20261102.md)，本轮科学结论见[CUDA配方分析](recipe_cuda_analysis_20261102.md)。
