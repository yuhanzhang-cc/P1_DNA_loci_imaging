# 标注工作流说明

这个目录提供一套适合你当前流程的 Fiji + TrackMate 半自动标注方案。原则是：

- 需要你判断的步骤继续手动做，例如选最佳 frame、选荧光 channel、调 TrackMate 参数。
- 可以批量和自动记录的步骤交给脚本处理，例如命名、保存、日志记录、从 TrackMate XML 反写 csv 和日志。

## 文件说明

- `fiji_prepare_annotation.ijm`
  作用：在 Fiji 中对“当前已打开的这一张图”执行准备工作，自动保存 `sample_N.tif`、启动 TrackMate、保存 `sample_N.png`、写入 sidecar 元数据。
- `annotation_tools.py`
  作用：在终端中执行批量辅助工作，包括：
  - 扫描源目录并生成按名称排序的 manifest
  - 查看当前进度和下一个 `sample_N`
  - 将 TrackMate XML 批量转换为 `data/raw/csv/sample_N.csv`
  - 追加正式日志 `data/raw/logs/annoation_record.csv`

## 输出目录

脚本会使用这些目录：

- `data/raw/tif/sample_N.tif`
- `data/raw/png/sample_N.png`
- `data/raw/csv/sample_N.csv`
- `data/raw/logs/trackmate_xml/sample_N.xml`
- `data/raw/logs/annotation_state/sample_N.meta.txt`
- `data/raw/logs/annoation_record.csv`

## 你和脚本如何配合

建议始终“一次只处理 1 个文件或 1 个 series”，不要在 Fiji 里同时打开很多待标注图。原因很简单：

- 你需要手动判断 frame、channel、z 投影，串行处理最稳妥。
- TrackMate 是交互式工具，同时开很多图很容易把保存路径、窗口名和日志记混。
- `.mvd2` 还会拆成多个 series，更适合逐个处理。

推荐操作方式如下。

### 第 0 步：可选，先生成待处理清单

如果你想先把移动硬盘里的源文件按名称顺序列出来，可以在终端运行：

```bash
python3 src/annotation_workflow/annotation_tools.py build-manifest /Volumes/你的硬盘/你的目录
```

这个命令不会打开 Fiji，只会生成一个清单 csv，方便你按顺序处理。

### 第 1 步：你自己在 Fiji 打开 1 个源文件

这一步需要你手动打开，不建议让脚本自动批量打开，原因是：

- `.nd2`、`.tiff`、`.mvd2` 在 Fiji/Bio-Formats 中的打开行为不完全一致。
- `.mvd2` 会展开成多个图像/series，自动批量打开反而更容易混乱。
- 你本来就需要先肉眼判断最佳 frame 和要保留的 channel。

所以建议流程是：

1. 在 Fiji 中手动打开当前要处理的 1 个源文件。
2. 如果是 `.mvd2`，Fiji 可能会出现多个图像窗口；此时只处理其中当前这 1 个 series。
3. 处理完这一张，再打开下一张，不要一次堆很多窗口。

### 第 2 步：你手动完成必须人工判断的内容

在 Fiji 里手动完成：

- 选最合适的 frame
- 确认荧光 channel
- 如果有 z stack，决定是否做 `Image > Stacks > Z Project > Max Intensity`

注意：

- 对 `.mvd2`，如果做了 Max Intensity 投影，日志中的 `source_file_name` 会自动加 `MAX_` 前缀。
- 对 `.mvd2`，`source_path` 统一记 `.mvd2` 文件路径，`source_file_name` 记当前 series 在 Fiji 里的图像标题。

### 第 3 步：运行 Fiji 宏

在当前图像窗口上运行：

- `src/annotation_workflow/fiji_prepare_annotation.ijm`

这个宏会做这些事情：

- 读取当前窗口的宽高像素和标定信息
- 让你确认 `source_file_name`、channel、是否做 z 投影
- 自动保存单帧导出图为 `data/raw/tif/sample_N.tif`
- 自动启动 TrackMate
- 在 TrackMate 结束后保存带圈结果图为 `data/raw/png/sample_N.png`
- 自动写一个 sidecar 文件，记录日志所需元数据

## 你在 Fiji 宏运行时要做什么

宏运行后，你只需要按界面提示做交互：

1. 选择这张图对应的原始源文件路径。
2. 确认弹窗里的：
   - `source_file_name`
   - `channel_index`
   - 是否应用 `Max Intensity z projection`
3. TrackMate 打开后，手动调：
   - `estimated object diameter`
   - `quality threshold`
4. 保存 TrackMate XML 到宏提示你的那个路径。
5. 关闭 TrackMate，宏会继续保存 png 和 sidecar。

## 第 4 步：批量收尾

你可以每做完 1 张就执行一次，也可以积累几张后统一执行：

```bash
python3 src/annotation_workflow/annotation_tools.py finalize
```

这个命令会自动：

- 读取 `data/raw/logs/annotation_state/sample_N.meta.txt`
- 解析 `data/raw/logs/trackmate_xml/sample_N.xml`
- 生成 `data/raw/csv/sample_N.csv`
- 追加一行到 `data/raw/logs/annoation_record.csv`
- 从 TrackMate XML 自动读取：
  - `estimated_object_diameter`
  - `quality_threshold`

## 新增日志字段

现在日志 csv 额外记录以下 4 列：

- `x_pixel`
  导出的 `sample_N.tif` 的宽度像素数
- `y_pixel`
  导出的 `sample_N.tif` 的高度像素数
- `micron_per_pixel_x`
  x 方向每个像素对应多少 micron
- `micron_per_pixel_y`
  y 方向每个像素对应多少 micron

这 4 个值由 Fiji 宏在导出 tif 时直接读取当前图像标定写入 sidecar，再由 Python 写入正式日志。

## 查看当前状态

```bash
python3 src/annotation_workflow/annotation_tools.py status
```

这个命令会告诉你：

- 下一个可用的 `sample_N`
- 还有多少 sidecar 没有 finalize
- 哪些记录缺少 TrackMate XML

## 关于 `.mvd2`

- `build-manifest` 只能列出 `.mvd2` 容器文件，不能直接列出其中每个 series 的真实名称。
- 真正的 `source_file_name` 仍然以 Fiji 当前图像标题为准。
- 如果你对某个 series 做了 Max Intensity 投影，宏会为 `.mvd2` 的 `source_file_name` 自动补 `MAX_` 前缀。

## 当前流程的边界

这套脚本没有尝试自动批量打开 Fiji 文件，也没有自动操作 TrackMate，因为这两部分都高度依赖你的人工判断。它的重点是：

- 减少重复命名和保存操作
- 自动补齐日志
- 避免 `tif/png/csv/log` 四处手抄信息
- 保持每一步都容易在 VS Code 和 Finder 中核对
