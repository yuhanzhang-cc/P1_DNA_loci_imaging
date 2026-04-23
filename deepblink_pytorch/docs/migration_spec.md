# 1. 关键 Shape 与坐标语义

## 1.1 输入图像

deepBlink 的训练与推理都以单通道二维图像为核心输入。网络实际接收的张量形状为：

- TensorFlow / Keras: `B x H x W x 1`
- 单张图像推理前的 numpy 形状: `H x W`

训练数据在 `SpotsDataset` 中被要求满足以下约束：

- 所有图像具有相同 shape
- 图像必须是正方形
- 边长必须是 2 的幂

推理阶段不要求输入满足上述约束。`deepblink.inference.predict()` 会先对图像做 z-score 标准化，再将图像 padding 到不小于原始边长的下一个 2 的幂，以适配网络下采样路径。

## 1.2 网络输出

网络输出是一个三通道预测矩阵，形状为：

- TensorFlow / Keras: `B x S_r x S_c x 3`
- 单张图像输出: `S_r x S_c x 3`

其中：

- `S_r = ceil(H / cell_size)`
- `S_c = ceil(W / cell_size)`

在 deepBlink 的标准训练流程中，训练图像是正方形，因此常见情况可写为 `S x S x 3`。

三个输出通道的语义固定为：

- 通道 0: `p`，该 cell 是否包含目标点的概率
- 通道 1: `r_rel`，目标点在该 cell 内部的相对行偏移
- 通道 2: `c_rel`，目标点在该 cell 内部的相对列偏移

网络最后一层为 3 通道 `Conv2D` 加 `sigmoid`，因此三个通道都位于 `[0, 1]` 范围内。

## 1.3 cell_size 的作用

`cell_size` 表示预测矩阵中一个网格 cell 对应原图中的多少个像素。它同时决定：

- 标签矩阵的空间分辨率
- 网络输出的空间分辨率
- 坐标编码与解码时的缩放关系

例如，当 `cell_size = 4` 时：

- 原图中每 `4 x 4` 像素对应预测矩阵中的一个 cell
- 输出矩阵尺寸约为输入尺寸的 `1 / 4`
- 相对偏移 `(r_rel, c_rel)` 表示目标点在该 `4 x 4` 区域内的位置

在网络实现中，`cell_size` 必须是 2 的幂。`unet.py` 中第二段 encoder 的深度由 `log2(cell_size)` 决定，因此模型结构直接依赖该参数。

## 1.4 坐标语义

deepBlink 内部统一使用 `(r, c)` 顺序表示坐标：

- `r`: row，对应图像的 y 轴
- `c`: column，对应图像的 x 轴

导出到 CSV 时会转换为更常见的显示习惯：

- 第一列 `y` 对应内部的 `r`
- 第二列 `x` 对应内部的 `c`

# 2. 标签编码协议（get_prediction_matrix）

## 2.1 输入格式

`get_prediction_matrix(coords, image_size, cell_size, size_c=None)` 的输入为：

- `coords`: `N x 2` 的坐标数组，按 `(r, c)` 排列
- `image_size`: 图像行方向长度，通常等于 `H`
- `cell_size`: 每个预测 cell 对应的像素尺寸
- `size_c`: 可选，图像列方向长度 `W`；若为空则默认使用正方形图像

每个坐标是原图像上的绝对坐标，而不是 cell 内相对坐标。

## 2.2 输出格式

输出是一个 shape 为 `nrow x ncol x 3` 的 numpy 数组：

- `nrow = ceil(H / cell_size)`
- `ncol = ceil(W / cell_size)`

第三维 3 个通道依次为：

- `[..., 0] = p`
- `[..., 1] = r_rel`
- `[..., 2] = c_rel`

其中：

- `p = 1` 表示该 cell 中存在一个点
- `r_rel` 表示点在 cell 内沿行方向的相对位置
- `c_rel` 表示点在 cell 内沿列方向的相对位置

## 2.3 编码规则

对每个绝对坐标 `(r, c)`：

1. 计算该点所属 cell
   - `cell_r = floor(r) // cell_size`
   - `cell_c = floor(c) // cell_size`
2. 将 cell 索引裁剪到有效范围内
3. 计算相对坐标
   - `r_rel = (r - cell_r * cell_size) / cell_size`
   - `c_rel = (c - cell_c * cell_size) / cell_size`
4. 将预测矩阵该位置写为
   - `(1, r_rel, c_rel)`

编码结果表示“每个 cell 最多承载一个点”。如果多个点落入同一个 cell，后写入的点会覆盖先前值。因此 `cell_size` 的选择会直接影响标签冲突概率。

## 2.4 训练前额外处理

在 `SpotsDataset.prepare_data()` 中，标签矩阵的概率通道会进一步做 smoothing：

- 正样本 cell: `p = smooth_factor`
- 负样本 cell: `p = 1 - smooth_factor`

默认 `smooth_factor = 1`，此时仍为严格的 0/1 标签。

## 2.5 解码协议

推理时，`get_coordinate_list(matrix, image_size, probability)` 将 `S_r x S_c x 3` 矩阵解码回 `(N, 2)` 坐标列表。

对满足 `matrix[..., 0] > probability` 的每个 cell：

1. 取出 cell 左上角绝对位置 `(cell_r * cell_size, cell_c * cell_size)`
2. 取出相对坐标 `(r_rel, c_rel)`
3. 计算绝对坐标
   - `r = cell_r * cell_size + r_rel * cell_size`
   - `c = cell_c * cell_size + c_rel * cell_size`

# 3. 损失函数公式

deepBlink 的训练损失是“分类损失 + 偏移回归损失”的组合。最常见配置为 `combined_dice_rmse`。

## 3.1 Dice score 与 Dice loss

令：

- `y_true^p` 为真实概率通道
- `y_pred^p` 为预测概率通道
- 将二者展平后记为向量

Dice score 定义为：

`Dice = (2 * sum(y_true^p * y_pred^p) + smooth) / (sum(y_true^p) + sum(y_pred^p) + smooth)`

其中 `smooth` 默认为 1，用于避免分母为 0。

Dice loss 定义为：

`L_dice = 1 - Dice`

## 3.2 RMSE 偏移损失

偏移回归只在真实存在目标点的位置计算。

令：

- `y_true^o = y_true[..., 1:3]`
- `y_pred^o = y_pred[..., 1:3]`

对于真实偏移为零的位置，deepBlink 使用掩码忽略这些位置对回归损失的贡献。设真实存在点的 cell 数为 `N_pos`，则：

`L_rmse = sqrt(sum(|| y_true^o - y_pred^o ||^2 over positive cells) / (N_pos + eps))`

这里每个正样本 cell 的平方误差由 `(r_rel, c_rel)` 两个分量共同构成。

## 3.3 组合损失

`combined_dice_rmse` 定义为：

`L_total = L_dice + 2 * L_rmse`

权重 2 是 deepBlink 原实现中的固定系数，用于提升偏移回归项的影响。

除了 `combined_dice_rmse` 外，原项目还支持：

- `combined_bce_rmse`
- `combined_f1_rmse`

但迁移阶段若以原主流程为目标，应优先复现 `combined_dice_rmse`。

# 4. 推理阈值流程

## 4.1 预处理

`deepblink.inference.predict(image, model, probability=None)` 的主要流程如下：

1. 对输入图像做 z-score 标准化
2. 将图像 padding 到下一个 2 的幂尺寸
3. 送入模型得到预测矩阵 `pred`

## 4.2 probability 参数的双重作用

`probability` 参数有两个作用：

1. 作为点筛选阈值
2. 决定输出中是否附加概率列

规则如下：

- 当 `probability is None` 时：
  - 使用默认阈值 `0.5` 解码点
  - 返回坐标 shape 为 `N x 2`，只包含 `(r, c)`
- 当 `probability` 为具体数值时：
  - 使用该值作为阈值解码点
  - 返回坐标 shape 为 `N x 3`，包含 `(r, c, p)`

## 4.3 点筛选逻辑

对预测矩阵中每个 cell：

- 若 `pred[cell_r, cell_c, 0] > threshold`，则认为该 cell 含有一个预测点
- 该点的位置由同一 cell 中的 `r_rel, c_rel` 解码得到

这里使用的是严格大于 `>`，而不是大于等于。

## 4.4 padding 区域移除

由于推理前可能对输入图像做了 padding，解码后的坐标中可能包含落在 padding 区域的点。deepBlink 会在解码后移除：

- `r > 原始图像高度`
- `c > 原始图像宽度`

的预测点，只保留原图范围内的坐标。

## 4.5 概率回填

当 `probability` 为显式数值时，deepBlink 会调用 `get_probabilities()`：

- 根据已解码坐标反查对应 cell
- 取该 cell 的 `p` 通道值
- 作为第三列附加到输出结果中

# 5. 输入/输出 CSV 格式

## 5.1 数据集创建阶段的输入 CSV

`deepblink cli create` 支持的标注 CSV 格式主要有两类：

### Fiji 点标注格式

要求包含列：

- `X`
- `Y`

读入后会重命名为内部坐标：

- `X -> c`
- `Y -> r`

### TrackMate 导出格式

要求包含列：

- `POSITION_X`
- `POSITION_Y`

读入后同样映射为：

- `POSITION_X -> c`
- `POSITION_Y -> r`

## 5.2 输入标注的单位

创建数据集时，输入标注默认可视为物理单位坐标。若提供 `pixel_size` 或从 tif 元数据中自动推断得到像素尺寸，则会按以下规则换算为像素坐标：

- `r_px = r / pixel_size_y`
- `c_px = c / pixel_size_x`

deepBlink 内部训练和推理都以像素坐标为标准。

## 5.3 预测输出 CSV

`deepblink cli predict` 导出的 CSV 以外部可读坐标为主，列顺序与内部 `(r, c)` 不同。

基础列为：

- 第一列: `y [px]` 或 `y [µm]`
- 第二列: `x [px]` 或 `x [µm]`

其中：

- `y` 对应内部的 `r`
- `x` 对应内部的 `c`

如果推理时提供了概率阈值参数，则附加：

- `p`

如果请求了积分强度统计，则还会附加：

- `i`

在多维图像批量预测场景中，还可能附加：

- `c`
- `t`
- `z`

若这些维度在整个文件中恒定不变，CLI 保存前会删除非唯一列。

## 5.4 本迁移项目的 TF 基线导出约定

为了便于 PyTorch 迁移阶段做逐文件对照，基线导出脚本统一输出三列：

- `y`
- `x`
- `p`

约定如下：

- 坐标单位统一为像素
- `y` 对应内部 `r`
- `x` 对应内部 `c`
- `p` 为触发阈值筛选后回填的概率值

该约定是迁移工程的基线格式，不要求完全复刻 deepBlink 原 CLI 的列命名细节。
