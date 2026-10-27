# 方法、实现与流程来源

查阅日期：2026-10-25。这里只列一手论文、作者官方实现和官方文档；列为候选不代表已安装、适配或完成比较。

## 点检测方法

1. Eichenberger, B. T., Zhan, Y., Rempfler, M., Giorgetti, L., & Chao, J. A. (2021). *deepBlink: threshold-independent detection and localization of diffraction-limited spots*. Nucleic Acids Research, 49(13), 7292–7297. [论文](https://doi.org/10.1093/nar/gkab546)；[作者实现](https://github.com/BBQuercus/deepBlink)。本仓库已有其PyTorch迁移，原TensorFlow与迁移版不计为两个独立方法族。
2. Dominguez Mantes, A., et al. (2025). *Spotiflow: accurate and efficient spot detection for fluorescence microscopy with deep stereographic flow regression*. Nature Methods. [论文](https://doi.org/10.1038/s41592-025-02662-x)；[官方仓库](https://github.com/weigertlab/spotiflow)；[自定义训练](https://weigertlab.org/spotiflow/train.html)；[安装与使用](https://weigertlab.org/spotiflow/)。用于外部专用点检测强基线候选；Windows构建依赖及现有XPU支持需要实际预检。
3. scikit-image官方：[Blob detection示例](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html)；[feature API](https://scikit-image.org/docs/stable/api/skimage.feature.html)。计划采用LoG/DoG作为传统参考，参数用开发数据选择，输出统一到原始图像坐标。

全分辨率U-Net heatmap+offset是计划中的本项目代表性基线，不宣称为上述文献中的另一官方方法，也不以其默认实现替代必要的调参与实现检查。

## 公开预训练权重补充

- Spotiflow：[预训练目录](https://weigertlab.org/spotiflow/pretrained.html)，核对general/fluo_live及其训练来源。
- U-FISH：[作者实现](https://github.com/u-science-lab/U-FISH)、[公开ONNX/PyTorch权重](https://huggingface.co/GangCaoLab/U-FISH/tree/main)、[快速入门](https://u-fish.readthedocs.io/en/latest/quickstart.html)；[论文](https://doi.org/10.1186/s13059-025-03736-x)。
- Piscis：[作者实现](https://github.com/zjniu/Piscis)、[官方推理与20251212模型](https://piscis.netlify.app/inference)、[权重仓库](https://huggingface.co/wniu/Piscis)；[论文](https://doi.org/10.1016/j.cels.2025.101448)。
- DeepSpot：[作者实现](https://github.com/cbib/DeepSpot)，仅作增强模型加固定检测器的备选完整流程，非首批必跑项。

这些资源的页面和加载入口已核对，实际权重下载、版本/hash固定、本机兼容和性能仍待预检，不将文献成绩当作本项目实测成绩。

## 统计与研究流程

规划沿用本机`statistical-analysis`技能：区分开发选择与确认性证据、报告效应量及不确定性、按照配对和相关结构确定采样单元、保留无效与失败结果。使用该流程不代表已完成统计检验，也不等于下列论文验证了本课题模型。

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。本次查阅记录为2026-09-02修订版，引用使用不带版本后缀的DOI。

## 本项目证据

- [220版报告](../../outputs/261024_clean220_random/最终汇报.md)：当前三种子性能、错误分析与校准信息。
- [同条件数据扩充报告](../../outputs/261025_clean220_control/最终汇报.md)：旧批次89张与合并176张的内部配对比较。
- [冻结划分](../../outputs/261024_clean220_random/frozen_split.csv)与[数据审计](../../outputs/261024_clean220_random/data_audit.json)：本轮数据、排除与尺寸/碰撞信息。
- [网络实现](../../deepblink_pytorch/deepblink_torch/networks/unet.py)与[SE实现](../../deepblink_pytorch/deepblink_torch/networks/_networks.py)：架构候选的实际代码起点。
- [旧上下文头结果](../../outputs/260630_context_head/deeploci_context/evaluation/summary/metrics_summary.json)：单次旧113实验记录，不是新220上的强比较结论。

本项目记录优先于旧会话中的临时运行状态；历史113、早期分源划分、当前合并随机划分和未来v2校准协议不能混成同一张无版本说明的比较表。

## 第二轮设计补充（2026-10-26核查）

- Wu, J., et al. *Single-Point Supervised High-Resolution Dynamic Network for Infrared Small Target Detection*. [作者预印本与方法全文](https://arxiv.org/html/2408.01976v2)。已有点监督、高分辨率交互和动态坐标融合，限制了“高分辨率＋注意力即创新”的主张；红外任务结果不是DNA loci效果证据。
- Zhang, G., et al. *Learning Dynamic Local Context Representations for Infrared Small Target Detection*. [作者全文](https://arxiv.org/html/2412.17401v1)。已有动态局部上下文、大核表示；R2需说明超出这些普通构件的可检验差别，不照搬其数据集成绩。
- Zhang, R. (2019). *Making Convolutional Networks Shift-Invariant Again*. [ICML/PMLR论文](https://proceedings.mlr.press/v97/zhang19a.html)。用于设计平移敏感性诊断；它不证明当前弱点漏检由池化造成，普通BlurPool也不是本项目新方法。
- PyTorch 2.11官方：[PixelUnshuffle](https://docs.pytorch.org/docs/2.11/generated/torch.nn.PixelUnshuffle.html)。R1复用空间/通道确定性重排；只是不在该步池化，不能声称整网无信息损失。
- Nibali, A., et al. (2018). *Numerical Coordinate Regression with Convolutional Neural Networks*. [作者预印本](https://arxiv.org/abs/1801.07372)。可微位置期望为既有思路；当前R1仍输出sigmoid offset残差，不将坐标变换本身作为创新。

Spotiflow的多尺度heatmap与stereographic flow也已通过[论文来源](https://doi.org/10.1038/s41592-025-02662-x)及[作者实现](https://github.com/weigertlab/spotiflow)重新核对。第二轮H0为本项目密集输出参考，不是Spotiflow官方复现。相对D/D2最近的局部对比、环形中心注意力等首轮资料仍保存在`outputs/next_stage_v2/08_innovation/related_work_matrix.csv`，本次未抹去原有文献边界。

执行期间继续核查更接近的构件级先例：

- Gao, Z., Wang, L., & Wu, G. (2019). *LIP: Local Importance-Based Pooling*. [ICCV作者论文](https://openaccess.thecvf.com/content_ICCV_2019/html/Gao_LIP_Local_Importance-Based_Pooling_ICCV_2019_paper.html)。已有输入条件化的局部重要性池化，不能把R1的软选择本身称作首次。
- Sunkara, R., & Luo, T. (2022). *No More Strided Convolutions or Pooling*. [作者预印本](https://arxiv.org/abs/2208.03641)。SPD-Conv已有space-to-depth加非步幅卷积，约束了S1/细节保存路径的创新主张。
- Liu, S., Huang, D., & Wang, Y. (2019). *Learning Spatial Fusion for Single-Shot Object Detection*. [作者预印本](https://arxiv.org/abs/1911.09516)。ASFF已有逐位置自适应多尺度融合；R2的尺度softmax不是新算子。
- Kobayashi, T. (2019). *Global Feature Guided Local Pooling*. [ICCV作者论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kobayashi_Global_Feature_Guided_Local_Pooling_ICCV_2019_paper.pdf)。已有更广特征信息引导局部池化的路线。
- Psomas, B., et al. (2023). *Keep It SimPool: Who Said Supervised Transformers Suffer from Attention Deficit?* [ICCV作者论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Psomas_Keep_It_SimPool_Who_Said_Supervised_Transformers_Suffer_from_Attention_ICCV_2023_paper.pdf)。已有单头query/key与softmax池化；R1需在具体检测路径、监督及机制对照上说明增量。

这些核查发生在结构筛查期间，不改变已启动训练的配置、预算或结果。它们提高创新主张的证据要求；实现和性能上涨本身均不能消除与既有工作的重合。
