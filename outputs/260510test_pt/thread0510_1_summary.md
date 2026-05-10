现状总结
  项目：D:\Projects\P1_DNA_loci_imaging\deepblink_pytorch，包名 deepblink_torch。目标是把原版 deepBlink
  的 spot localization 工作流迁移到 PyTorch，并解决 TrackMate 微米坐标与模型像素坐标之间的单位体系问
  题。

  已经达成的单位体系原则：

  - 模型内部永远使用像素坐标。
  - create 阶段负责把 TrackMate CSV 中的微米坐标转换成像素坐标。
  - train / loss / decode / overlay 都基于像素坐标。
  - predict 默认输出像素坐标；可选输出微米坐标用于生物学分析。
  - 每张图像的 pixel size 随 .npz 保存，不依赖外部查表。
  - 从 TIF 元数据读取不到 pixel size 时，使用 1.0 um/px 并警告。

  已经实现的部分
  对像素/微米的转换已实现“只改头尾”的单位体系改造，未改 unet / losses / data_protocol / training 主体，在模型内部只用像素单位的坐标进行计算和学习：
  
  现在的行为是：

  - create 先从 tif 读 pixel size，读到就用真实值，读不到就默认 1.0 um/px 并警告。
  - create 会把 TrackMate 的微米坐标换成像素坐标后再写入 npz。
  - npz 里新增了 pixel_size_x_*、pixel_size_y_*、pixel_size_source_*，SpotsDataset 会读这些字段，但训练
    仍只用像素坐标，不受影响。
  - predict 默认仍输出像素坐标；--output-unit micron 会按当前图像的 pixel size 转回微米。读不到 pixel
    size 时同样回退到 1.0 并警告。

具体的代码改动是：
  - 新增 deepblink_torch/utils/pixel_size.py
      - read_pixel_size_from_image()
      - 支持 ImageJ TIFF、OME metadata、标准 TIFF XResolution / YResolution / ResolutionUnit
      - 读不到时可通过 read_pixel_size_or_default() fallback 到 1.0 um/px
  - cli/create.py
      - 读取每张 TIF 的 pixel size
      - 将 CSV 微米坐标转成像素坐标：y_px = y_um / pixel_size_y，x_px = x_um / pixel_size_x
      - .npz 中继续保存标准字段：x_train/y_train/x_valid/y_valid/x_test/y_test
      - 新增 metadata 字段：
          - pixel_size_x_train/y/source
          - pixel_size_x_valid/y/source
          - pixel_size_x_test/y/source
  - datasets/spots.py
      - 能读取新增 pixel size metadata
      - 训练仍只使用标准图像和像素坐标标签
      - 额外字段只做 traceability，不参与训练
  - cli/predict.py
      - 默认输出像素坐标
      - 新增 --output-unit {px,micron}
      - --output-unit micron 时用当前预测图像的 pixel size 把像素坐标乘回微米
  - 已有验证：
      - sample_4.tif 可读到约 0.133 um/px
      - sample_100.tif fallback 到 1.0
      - create 坐标转换 smoke test 通过
      - predict micron conversion smoke test 通过

  原版 deepBlink 协议对齐结论
  对照本机安装的原版 deepblink 0.1.4 Python 源码，以及 GitHub 仓库 BBQuercus/deepBlink：

  已对齐：

  - U-Net 主体结构基本对齐：
      - 原版有第一段 encoder、decoder、第二段与 cell_size 绑定的 encoder
      - PyTorch 版 UNetDeepBlink 也实现了这套结构
      - 默认 cell_size=4, ndown=2, block="convolutional" 路径基本等价
  - 输出语义对齐：
      - 输出 shape：(B, H/4, W/4, 3)，channel-last
      - 三通道语义：[probability, r_rel, c_rel]
  - 标签编码 / 解码对齐：
      - get_prediction_matrix
      - absolute_coordinate
      - get_coordinate_list
      - 使用 floor(coord) // cell_size
      - 坐标落在 cell 边界时归属右/下 cell
      - 同一个 cell 多个点时后写覆盖先写，和原版一致
  - loss 对齐：
      - dice_loss + 2 * rmse_loss
      - RMSE 只在正样本点上计算
      - 背景 cell 不参与 offset regression
  - .npz 结构保持兼容：
      - 标准字段未破坏
      - 新增 pixel size 字段为附加 metadata
      - 主训练流程可忽略新增字段

  已知偏差 / 风险
  当前不是 bit-for-bit 完全复刻原版，主要偏差如下：

  1. Inference padding 偏差

  原版 deepblink.inference.predict：

  - 对 height 和 width 分别 pad 到各自的 next power of 2
  - 使用 np.pad(..., mode="reflect")
  - 不强制 pad 成正方形

  当前 deepblink_torch.cli.predict：

  - pad 成正方形：target = next_power(max(H, W))
  - 使用 zero padding
  - 这会改变边缘区域上下文，可能影响靠近右边界/下边界的预测

  优先级：中高。若追求原版行为，应该优先修这个。

  2. 越界点移除边界条件轻微偏差

  原版删除条件是：

  - coords[0] > image.shape[0] 或 coords[1] > image.shape[1]

  当前 PyTorch 版保留条件是：

  - coords[:, 0] < orig_h 且 coords[:, 1] < orig_w

  当前逻辑更符合 Python 像素索引范围，但与原版在恰好等于 H/W 的边界点处理上不完全一致。

  优先级：低到中。生物学上通常影响很小，但如果要严格复刻原版要处理。

  3. 非默认 block 路径存在 bug

  deepblink_torch/networks/unet.py 中 _DeepBlinkBlock.__init__() 里有 del l2，但后面
  inception_block(..., l2_regularizer=l2) 还会引用 l2。

  影响：

  - 默认 block="convolutional" 不受影响
  - block="inception" 会报错
  - block="residual" 不直接用 l2，但这段写法仍然不干净

  优先级：中。若近期只用默认 convolutional，可以暂不处理；若要开放 block 选项，应修。

  4. cell_size 边缘行为

  默认 cell_size=4 与原版主用路径对齐。更一般的 cell_size 理论上支持 power of 2，但以下行为需谨慎：

  - train.py 目前硬编码 cell_size=4
  - checkpoint 里保存 cell_size=4
  - 预测从 checkpoint 读取 cell_size
  - 非 4 的完整训练/预测路径没有充分验证

  优先级：低。当前 DNA spot 任务建议先固定 cell_size=4，不要扩展复杂度。

  5. 训练增强未完整复刻原版

  原版 SpotsModel 中有 augmentation pipeline，例如 flip、rotate、translate、illumination、Gaussian
  noise 等。当前 PyTorch CLI training 是简化训练流程。

  优先级：中。是否补齐取决于当前训练泛化情况。

  现在不建议再改的地方
  短期内不建议动这些模块，除非有明确实验问题：

  - data_protocol.py
      - 编码/解码协议已经对齐，且它是全系统坐标语义核心。不要轻易改。
  - losses.py
      - 当前实现符合原版 dice + 2*rmse，先保留作为 baseline。
  - UNetDeepBlink 默认 convolutional 主体
      - 当前默认结构和原版基本一致，先作为 baseline 跑通。
  - .npz 标准字段
      - 不要破坏 x_train/y_train/x_valid/y_valid/x_test/y_test
      - pixel size metadata 应保持附加字段，不应嵌入或替代主标签。
  - 单位体系
      - 不要再让训练流程混入微米坐标。
      - 微米只出现在 create 输入转换和 predict 可选输出转换。

  优化优先级
  建议按这个顺序处理：

  1. 修 inference padding 到更接近原版
      - 从 zero square padding 改为 reflect padding
      - height/width 分别 pad 到 next power of 2
      - 解码时正确传入 padded image size / cell size
      - 这是最可能影响预测边缘质量的协议偏差
  2. 修 _DeepBlinkBlock 的 del l2 bug
      - 简单修复，风险低
      - 让 block="inception" 至少不会直接崩
  3. 给 inference padding / decode / boundary 行为加单元测试
      - 用小矩阵固定输入验证 decode
      - 用小图验证 padding shape 和边界过滤
  4. 根据训练表现决定是否补 augmentation
      - 如果训练集小、泛化差，再考虑复刻原版增强
      - 不要在 baseline 未稳定前大改训练策略
  5. 再考虑输出 CSV 扩展
      - 例如同时输出 y_px/x_px/y_um/x_um/p
      - 当前已有 --output-unit，短期够用

  未来面向 DNA 成像场景可大改的位置如果未来目标从“复刻 deepBlink”转向“适合 DNA loci / 单分子斑点检测的新模型”，可以重点发挥主观能动性的位置：

  1. 网络结构

  可改位置：

  - deepblink_torch/networks/unet.py
  - deepblink_torch/networks/_networks.py

  可探索方向：

  - Attention U-Net
  - SE / CBAM / spatial attention 加强弱斑点响应
  - Feature Pyramid Network / U-Net++，改善多尺度斑点与背景结构区分
  - 更浅或更轻的 backbone，避免对小训练集过拟合
  - 多尺度输出 head，同时预测小斑点和较大干扰结构

  2. Loss function

  可改位置：

  - deepblink_torch/losses.py

  可探索方向：

  - focal loss / weighted BCE 改善正负样本极不平衡
  - heatmap Gaussian target 替代单 cell occupancy
  - offset loss 从 RMSE 改 SmoothL1 / Huber
  - 引入 false positive penalty，针对核仁、亮背景、团块干扰
  - 根据 spot 距离或局部背景动态加权

  3. 标签表示 / 编解码协议

  可改位置：

  - deepblink_torch/data_protocol.py

  但注意：这属于大改，不再保持 deepBlink 原协议。

  可探索方向：

  - 从 cell classification + offset 改为 full-resolution / low-resolution heatmap
  - 用 Gaussian blob label 表达 spot 面积和定位不确定性
  - 支持一个 cell 多个 spot，避免当前“同 cell 后写覆盖先写”
  - 引入 spot radius / intensity / area 作为辅助 target

  4. DNA 成像特异性后处理

  可新增模块，不建议塞进 data_protocol.py：

  - spot candidate filtering
  - morphology / connected components
  - intensity profile fitting
  - 排除核仁或大面积亮斑
  - 根据 spot 面积、圆度、局部背景、SNR 过滤
  - 与 ImageJ/TrackMate 结果做校准对照

  5. 输入通道和预处理

  可改位置：

  - dataset / create / predict 的图像读取与 normalization 处

  可探索方向：

  - 局部背景扣除
  - rolling-ball / top-hat preprocessing
  - 多通道输入，例如 DNA channel + nuclear marker
  - patch-based training/predict，支持非 2 的幂、非方形、大图
  - per-image normalization 改为 robust normalization，如 percentile clipping

  6. Evaluation

  建议新增独立评估脚本，而不是混入训练：

  - matching by distance threshold
  - precision / recall / F1
  - localization error distribution
  - per-image FP/FN 分析
  - 对核仁区域、大亮斑区域单独统计误检

  一句话定位
  当前 deepblink_torch 最适合作为“原版 deepBlink 协议兼容 baseline”：单位体系已经理顺，训练标签和模型输出都是像素坐标，核心编解码/loss/U-Net 默认路径基本对齐。下一步若继续做兼容性，应优先修 inference padding 和非默认 block bug；若转向 DNA 成像专用模型，则应把大改集中在网络结构、loss、标签表达和后处理，而不要再反复扰动已经稳定的单位体系和基础协议。