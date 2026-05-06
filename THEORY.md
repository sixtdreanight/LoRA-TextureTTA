# DeepfakeBench — 理论文档

本文档记录项目中所有用到的理论、公式和算法，包含详细讲解和代码位置。

---

## 目录

1. [损失函数](#1-损失函数)
2. [网络架构](#2-网络架构)
3. [检测器与微调方法](#3-检测器与微调方法)
4. [优化策略](#4-优化策略)
5. [数据增强与推理策略](#5-数据增强与推理策略)
6. [评估指标](#6-评估指标)

---

## 1. 损失函数

### 1.1 BCELoss（二元交叉熵损失）

- **文件**: `DeepfakeBench/training/loss/bce_loss.py`
- **注册名**: `bce`
- **论文**: 无（标准损失函数）

**公式**：

$$L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

**说明**: 用于二分类任务的标准二元交叉熵损失，衡量预测概率与真实标签之间的差异。

---

### 1.2 CrossEntropyLoss（交叉熵损失）

- **文件**: `DeepfakeBench/training/loss/cross_entropy_loss.py`
- **注册名**: `cross_entropy`
- **论文**: 无（标准损失函数）

**公式**：

$$L = -\frac{1}{N}\sum_{i=1}^{N} \log\left(\frac{e^{z_{i,y_i}}}{\sum_{j} e^{z_{i,j}}}\right)$$

**说明**: 多分类交叉熵损失，使用 PyTorch 的 `nn.CrossEntropyLoss()`。

---

### 1.3 AM-Softmax（加性边际 Softmax）

- **文件**: `DeepfakeBench/training/loss/am_softmax.py`
- **注册名**: `am_softmax`, `am_softmax_ohem`
- **论文**: Wang et al., "Additive Margin Softmax for Face Verification", arXiv:1801.05599, 2018

**公式**：

标准 AM-Softmax（`margin_type='cos'`）:

$$L = -\frac{1}{N}\sum_i \log\frac{e^{s \cdot (\cos\theta_{y_i} - m)}}{e^{s \cdot (\cos\theta_{y_i} - m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$

ArcFace 变体（`margin_type='arc'`）:

$$L = -\frac{1}{N}\sum_i \log\frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$

其中 $\theta$ 为特征向量与分类权重之间的夹角，$s$ 为缩放因子，$m$ 为边际。

OHEM 版本在此基础上只对损失值最高的 `ratio × N` 个难样本进行反向传播，源自 Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining", CVPR 2016。

**关键参数**:
- `margin_type`: `'cos'`（AM-Softmax）或 `'arc'`（ArcFace）
- `s`: 缩放因子
- `m`: 边际值
- `gamma`: 焦点损失缩放（可选）
- `t`: 自适应调节参数（可选）
- `ratio`: OHEM 保留比例

---

### 1.4 对比正则化损失（Contrastive Regularization）

- **文件**: `DeepfakeBench/training/loss/contrastive_regularization.py`
- **注册名**: `contrastive_regularization`
- **论文**: 基于对比表示学习理论

**公式**（三元组形式）:

$$L = \frac{1}{N}\sum_{i} \max\left(0, d(f_i^a, f_i^p) - d(f_i^a, f_i^n) + \alpha\right)$$

其中 $d(\cdot,\cdot)$ 为欧氏距离，$f^a$ 为锚点特征，$f^p$ 为正样本特征，$f^n$ 为负样本特征，$\alpha$ 为边际。

**说明**: 在"共同"（shared）与"特定"（specific）特征之间计算对称对比损失，使同类样本靠近、异类样本远离。

---

### 1.5 胶囊损失（CapsuleLoss）

- **文件**: `DeepfakeBench/training/loss/capsule_loss.py`
- **注册名**: `capsule_loss`
- **论文**: Sabour et al., "Dynamic Routing Between Capsules", NeurIPS 2017

**公式**：

$$L = \sum_{k=0}^{K-1} \text{CE}(\text{out}[:, k, :])$$

**说明**: 针对胶囊网络的多输出向量，对每个胶囊输出分别计算交叉熵损失后求和。

---

### 1.6 一致性损失（Consistency Loss）

- **文件**: `DeepfakeBench/training/loss/consistency_loss.py`
- **注册名**: `consistency_loss`
- **论文**: 半监督一致性正则化（通用框架）

**公式**：

$$L = L_{CE} + \lambda \cdot \text{MSE}(\text{cos\_sim}(\hat{z}_1, \hat{z}_2), 1.0)$$

**说明**: 将批次分成两半，计算归一化特征之间的余弦相似度，并强制该相似度接近 1（通过 MSE），同时加入标准交叉熵损失。用于半监督学习的自一致性约束。

---

### 1.7 ID Loss（身份保持损失）

- **文件**: `DeepfakeBench/training/loss/id_loss.py`
- **注册名**: `id_loss`
- **论文**: 类似 ArcFace（Deng et al., arXiv:1801.07698）

**公式**：

$$\theta = \arccos(\text{cos\_sim}(f_1, f_2))$$
$$L = 1 - \cos(\theta + m)$$

**说明**: 计算两个特征向量的余弦相似度并转换为角度 $\theta$，计算 $1 - \cos(\theta + m)$，在角度边际上强制身份相似性。

---

### 1.8 VGG 感知损失（Perceptual VGG Loss）

- **文件**: `DeepfakeBench/training/loss/vgg_loss.py`
- **注册名**: `vgg_loss`
- **论文**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution", ECCV 2016, arXiv:1603.08155

**公式**：

$$L_{VGG} = \frac{1}{C_f H_f W_f} \sum_{c,h,w} \|\phi_f(I_{pred})_{c,h,w} - \phi_f(I_{target})_{c,h,w}\|_2^2$$

**说明**: 在预训练 VGG16/VGG19 的选定层（默认 `relu2_2`）特征空间中，计算预测图像与目标图像之间的 MSE 距离。可选的 TVLoss（总变分损失）来自 Mahendran & Vedaldi, "Understanding Deep Image Representations by Inverting Them", CVPR 2015。

---

### 1.9 JS 散度损失

- **文件**: `DeepfakeBench/training/loss/js_loss.py`
- **注册名**: `jsloss`
- **论文**: Lin, J., "Divergence Measures Based on the Shannon Entropy", IEEE Trans. Info. Theory, 1991

**公式**：

$$JS(P\|Q) = \frac{1}{2}KL(P\|M) + \frac{1}{2}KL(Q\|M)$$
$$M = \frac{P + Q}{2}$$

**说明**: Jensen-Shannon 散度是对称的 KL 散度变体，范围 $[0, \log 2]$，比 KL 散度更稳定。

---

### 1.10 分类+分割损失（ClassNseg Loss）

- **文件**: `DeepfakeBench/training/loss/classNseg_loss.py`
- **注册名**: `classNseg_loss`

**组成**（三项联合）:
1. **ActivationLoss**: 强制特征通道特定于真实/伪造
2. **SegmentationLoss**: 预测分割掩码上的交叉熵
3. **ReconstructionLoss**: 重建图像上的 MSE

---

## 2. 网络架构

### 2.1 Xception

- **文件**: `DeepfakeBench/training/networks/xception.py`
- **注册名**: `xception`
- **论文**: Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions", CVPR 2017, arXiv:1610.02357

**核心思想**: 使用深度可分离卷积（depthwise separable convolution）替代标准卷积，将空间相关性和通道相关性完全解耦。

**架构组成**:
- **入口流（Entry Flow）**: 3个块，逐步将通道从 3→32→64→128→256→728
- **中间流（Middle Flow）**: 8个重复的 728→728 块
- **出口流（Exit Flow）**: 728→1024→1536→2048

---

### 2.2 ResNet 系列

- **文件**: `DeepfakeBench/training/networks/resnet.py`, `resnet34.py`
- **注册名**: `resnet34`
- **论文**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016, arXiv:1512.03385

**核心思想——残差学习**:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

其中 $\mathcal{F}$ 为残差映射，$\mathbf{x}$ 为恒等映射。

**支持变体**: ResNet-18, 34, 50, 101, 152；IRBlock 变体（带 SEBlock + PReLU）、AdaIN Block 变体。

---

### 2.3 IResNet（改进残差网络）

- **文件**: `DeepfakeBench/training/networks/iresnet.py`
- **论文**: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", arXiv:1801.07698

**核心改进**: 使用 IBasicBlock（BN→Conv→BN→PReLU→Conv→BN），移除最后一个 ReLU 避免激活截断。

**支持变体**: iresnet18, 34, 50, 100, 200

---

### 2.4 MesoNet 系列

- **文件**: `DeepfakeBench/training/networks/mesonet.py`
- **注册名**: `meso4`, `meso4Inception`
- **论文**: Afchar et al., "MesoNet: a Compact Facial Video Forgery Detection Network", WIFS 2018

**核心思想**: 针对深度伪造检测的浅层网络，少量层足以捕获中观级（mesoscopic）伪造痕迹。

**Meso4 结构**: Conv(8)→Pool→Conv(8)→Pool→Conv(16)→Pool→Conv(16)→Pool→FC(16)→FC(2)

**MesoInception4**: 在 Meso4 基础上引入 Inception 模块（4条并行扩张路径 dil=1,2,3）。

---

### 2.5 EfficientNet-B4

- **文件**: `DeepfakeBench/training/networks/efficientnetb4.py`
- **注册名**: `efficientnetb4`
- **论文**: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019, arXiv:1905.11946

**核心思想——复合缩放**: 使用复合系数 $\phi$ 统一缩放深度、宽度和分辨率：

$$d = \alpha^\phi,\quad w = \beta^\phi,\quad r = \gamma^\phi$$
$$\text{s.t. } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$$

---

### 2.6 HRNet（高分辨率网络）

- **文件**: `DeepfakeBench/training/networks/cls_hrnet.py`
- **论文**: Wang et al., "Deep High-Resolution Representation Learning for Visual Recognition", TPAMI 2020, arXiv:1908.07919

**核心思想**: 全程保持高分辨率表示，同时并行维护多分辨率流并进行重复多分辨率融合。

---

### 2.7 AdaFace / InsightFace IR

- **文件**: `DeepfakeBench/training/networks/adaface.py`
- **论文**: Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition", CVPR 2022, arXiv:2204.00964

**核心思想**: 基于图像质量自适应调整边际，低质量图像使用较小边际。使用 GNAP（全局范数感知池化）和 GDC（全局深度卷积）。

---

## 3. 检测器与微调方法

### 3.1 EffortDetector

- **文件**: `DeepfakeBench/training/detectors/effort_detector.py`
- **注册名**: `effort`

**核心设计**:

1. **CLIP 视觉编码器**（冻结）
   - **论文**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021, arXiv:2103.00020
   - 使用 ViT-L/14 作为视觉骨干，不参与梯度更新

2. **LoRA 微调**（Low-Rank Adaptation）
   - **论文**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022, arXiv:2106.09685
   - 对 CLIP 的注意力投影层（q_proj, k_proj, v_proj, out_proj）应用低秩适配器，秩 r=4
   - **核心公式**: $W' = W + \Delta W = W + BA$，其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$
   - 只训练低秩矩阵 A 和 B，冻结原始权重 W

3. **不对称中心损失（Margin Loss）**
   - 将真实样本特征拉向可学习中心 $c_R$，将伪造样本特征推离超过边际 $m$
   - $$L_{margin} = \frac{1}{N}\sum_i \left[y_i \cdot \|f_i - c_R\|^2 + (1-y_i) \cdot \max(0, m - \|f_i - c_R\|)^2\right]$$
   - 特征和中心均经过 L2 归一化

4. **自适应阈值（OWTTT 风格）**
   - 基于最近预测的滑动窗口计算动态决策阈值（搜索范围 0.1-0.9）
   - 最小化加权类内方差减去间隙惩罚

5. **纹理感知聚合（TAA）**
   - 使用 Laplacian 方差作为清晰度度量，提取多裁剪块
   - 通过伽马加权 softmax 聚合纹理最丰富和最平滑的块
   - $$S(I) = \beta \cdot s_{full} + (1-\beta) \cdot \sum_j w_j \cdot s_j, \quad w_j = \frac{t_j^\gamma}{\sum_k t_k^\gamma}$$

---

## 4. 优化策略

### 4.1 PCGrad（梯度手术）

- **文件**: `DeepfakeBench/training/optimizor/pcgrad.py` / `DeepfakeBench/Pytorch-PCGrad-master/`
- **论文**: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020, arXiv:2001.06782

**核心思想**: 当两个任务的梯度方向冲突（点积 < 0）时，将每个梯度投影到另一个梯度的法平面上，消除冲突分量：

$$g_i' = g_i - \frac{g_i \cdot g_j}{\|g_j\|^2} g_j \quad \text{if } g_i \cdot g_j < 0$$

**说明**: 以随机顺序处理所有任务对的梯度，减少顺序偏差。

---

### 4.2 SAM（锐度感知最小化）

- **文件**: `DeepfakeBench/training/optimizor/SAM.py`
- **论文**: Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021, arXiv:2010.01412

**核心思想**: 寻找损失景观中不仅损失低、而且邻域平坦的参数点，提高泛化性能。

**两步更新**:

1. 扰动步：$\epsilon = \rho \cdot \frac{\nabla_w L(w)}{\|\nabla_w L(w)\|}$，计算 $L(w + \epsilon)$
2. 更新步：$w = w - \eta \cdot \nabla_w L(w + \epsilon)$

---

### 4.3 不对称 Mixup

- **文件**: `DeepfakeBench/training/trainer/trainer.py`
- **论文**: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018, arXiv:1710.09412

**核心公式**:

标准 Mixup：
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

不对称变体（真实-伪造混合）：
$$y_{mixed} = 1 - (\text{real\_prop})^\gamma$$

**说明**: 对于真实与伪造样本的混合，积极将混合样本推入伪造类，使用不对称软标签。

**Hardest-K Mixup**: 对每个真实样本生成 K 个候选混合，通过无梯度前向传播选择损失最高的（最困难的）候选。

---

### 4.4 SWA（随机权重平均）

- **论文**: Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization", UAI 2018, arXiv:1803.05407

**核心思想**: 在训练后期对多个 epoch 的模型权重进行平均，寻找更平坦的极小值以提升泛化：

$$w_{SWA} = \frac{1}{N_{models}} \sum_i w_i$$

---

## 5. 数据增强与推理策略

### 5.1 数据增强

- **文件**: `DeepfakeBench/training/dataset/albu.py`
- **使用库**: Albumentations

**增强列表**: HorizontalFlip, RandomBrightnessContrast, HueSaturationValue, ImageCompression, GaussNoise, MotionBlur, CLAHE, ChannelShuffle, Cutout, RandomGamma, GlassBlur

---

### 5.2 多裁剪推理（TAA — 纹理感知聚合）

- **文件**: 在 `effort_detector.py` 中实现

**流程**:
1. 使用滑动窗口提取多个裁剪块
2. 计算每个块的 Laplacian 方差（清晰度度量）
3. 选择纹理最丰富（`texture_Kr` 个）和最平滑（`texture_Ks` 个）的块
4. 通过伽马加权 softmax 聚合预测：$w_j = t_j^\gamma / \sum_k t_k^\gamma$
5. 最终分数：$S(I) = \beta \cdot s_{full} + (1-\beta) \cdot \sum_j w_j \cdot s_j$

当纹理分数不可用时，回退到关注置信度聚合（选择 `argmax(|prob - 0.5|)` 的最自信块）。

---

## 6. 评估指标

- **文件**: `DeepfakeBench/training/metrics/base_metrics_class.py`

### 6.1 ACC（准确率）

$$ACC = \frac{TP + TN}{TP + TN + FP + FN}$$

### 6.2 AUC（ROC 曲线下面积）

通过变化分类阈值，绘制 TPR vs FPR 曲线，计算曲线下面积。AUC=1 为完美分类，AUC=0.5 为随机猜测。

### 6.3 EER（等错误率）

FPR = FNR 时的错误率，常用于人脸验证和伪造检测的阈值无关评估。

### 6.4 AP（平均精度）

Precision-Recall 曲线下面积，适用于类别不平衡场景。

---

> **更新规则**: 每次在项目中新增理论或算法时，必须同步更新本文档，添加对应章节。
