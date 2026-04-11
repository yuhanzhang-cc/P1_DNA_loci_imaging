# 标注工作流说明

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

## 和脚本如何配合

手动判断 frame、channel、z 投影，以及Trackmate取点。


### 第 1 步：手动在 Fiji 打开 1 个源文件


1. 在 Fiji 中手动打开当前要处理的 1 个源文件。
2. 如果是 `.mvd2`，Fiji 可能会出现多个图像窗口；此时只处理其中当前这 1 个 series。

### 第 2 步：人工判断

在 Fiji 里手动完成：

- 选最合适的 frame
- 确认荧光 channel
- 如果有 z stack，决定是否做 `Image > Stacks > Z Project > Max Intensity`

注意：

- 对 `.mvd2`，如果做了 Max Intensity 投影，日志中的 `source_file_name` 会自动加 `MAX_` 前缀。
- 对 `.mvd2`，`source_path` 统一记 `.mvd2` 文件路径，`source_file_name` 记当前 series 在 Fiji 里的图像标题。

### 第 3 步：运行 Fiji 宏

在当前图像窗口上运行：

- `src/annotation_workflow/fiji_prepare_annotation.ijm`，快捷键M

这个宏会做这些事情：

- 读取当前窗口的宽高像素和标定信息
- 在对话框里确认 `source_path`、`source_file_name`、channel、是否做 z 投影
- 自动保存单帧导出图为 `data/raw/tif/sample_N.tif`
- 自动启动 TrackMate
- 在 TrackMate 结束后保存带圈结果图为 `data/raw/png/sample_N.png`
- 自动写一个 sidecar 文件，记录日志所需元数据

宏运行后，按界面提示做交互：

1. 确认弹窗里的：
   - `source_path`
   - `source_file_name`
   - `channel_index`
   - 是否应用 `Max Intensity z projection`
2. TrackMate 打开后，手动调：
   - `estimated object diameter`
   - `quality threshold`
3. 手动保存 TrackMate XML 到宏提示的路径。
4. 关闭 TrackMate。
5. 回到宏的确认框，按 `Enter` 或点击 `OK`，宏继续保存 png 和 sidecar。

### 第 4 步：批量收尾


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

### 查看当前状态

```bash
python3 src/annotation_workflow/annotation_tools.py status
```

- 下一个可用的 `sample_N`
- 还有多少 sidecar 没有 finalize
- 哪些记录缺少 TrackMate XML

