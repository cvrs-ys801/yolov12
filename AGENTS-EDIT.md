# YOLOv12 DMMA/ ECA 改进报告（2025-12-15）

## 背景
用户项目基于 YOLOv12，核心创新为 Difference Mask Mixed Attention (DMMA) 与 ECA 通道门控，用于小型舰船目标检测。此前实现为单尺度窗口 DMMA + 固定核 ECA。

## 发现的问题
- 单尺度窗口（固定 window_size）难兼顾不同尺寸舰船与背景上下文，shift 仅作用于单一窗口层，空间适应性不足。
- ECA 核大小固定为 3，且仅使用均值池化，通道权重缺乏对极值与通道宽度的自适应，易抑制细小目标。
- DMMA 掩码对注意力的影响为固定权重，缺少可学习温度/强度，且 dropout 位于 softmax 之前，可能带来分布不稳定。

## 改进方案

1. **多尺度 DMMA (MS-DMMA)**  
   - 新增 `MSDMMALayer`，并在 `C2fDMMA` 中可通过 `window_size` 传入列表自动启用多尺度并行窗口（如 `[4, 8]` 或 `[8, 16]`），Softmax 门控融合输出，支持奇偶层交替 shift。  
2. **自适应 ECA**  
   - ECA 升级为动态核长（ECA 公式），加入最大池化分支与可学习缩放，强化显著通道并保留细节。  
3. **可学习注意力温度与掩码增益**  
   - 为 DMMA 引入 head 级温度和掩码缩放参数，将 dropout 后移到 softmax 之后，稳定梯度并让模型自调掩码强度。

## 代码改动位置
- `ultralytics/nn/modules/transformer.py`  
  - 新增 `_auto_eca_kernel`、`MSDMMALayer`。  
  - `DifferenceMaskAttention` 增加 head 温度与掩码缩放，调整 softmax/dropout 顺序。  
  - `DMMAChannelAttention` 改为动态核 + Avg/Max 双分支 + learnable scale。  
- `ultralytics/nn/modules/block.py`  
  - `C2fDMMA` 支持多尺度窗口列表，自动选择 `MSDMMALayer`，保持原单尺度兼容。  
- 新增模型配置 `ultralytics/cfg/models/v12/yolov12-dmma-ms.yaml`：P4/P5 主干与 P3/P4/P5 头均采用双窗口 MS-DMMA 版本。

## 使用方法
- 直接训练多尺度版本：  
  ```bash
  yolo train model=ultralytics/cfg/models/v12/yolov12-dmma-ms.yaml data=your.yaml
  ```
- 若想在现有配置中试验，只需将 `C2fDMMA` 第二个参数写为列表，例如 `C2fDMMA, [256, [4,8], 8, False, 1.5]`，无需额外代码修改。

## 预期收益
- 多尺度窗口让小舰船与背景/海面纹理同时被关注，提升小目标召回。  
- 自适应 ECA 在窄通道场景下自调整核长，并结合极值池化，增强显著通道对比度。  
- 可学习温度/掩码强度降低过度抑制或放大，训练更稳健。

## 验证建议
1. 安装 `torch` 后进行前向烟雾测试（未在当前环境执行）：  
   ```bash
   python - <<'PY'
   import torch
   from ultralytics.nn.modules.block import C2fDMMA
   m=C2fDMMA(512,512,n=2,window_size=[4,8],num_heads=8,shift=True,mlp_ratio=2.0)
   x=torch.randn(1,512,20,20)
   y=m(x)
   print(y.shape)
   PY
   ```
2. 训练对比：`yolov12-dmma.yaml` vs `yolov12-dmma-ms.yaml`，记录 mAP50-95、小目标召回、推理耗时。  
3. 部署受限设备可仅在 P3 使用多尺度窗口，P4/P5 保持单尺度以控算力。

## 兼容性与风险
- 向后兼容：单尺度用法保持不变。  
- 额外参数极少（温度/掩码增益、ECA scale），对模型大小影响可忽略。  
- 多尺度分支增加计算量；在嵌入式场景可调小窗口或减少分支。

---

# YOLOv12-DMMA-P2-Advanced 改进报告（2025-12-29）

## 背景
基于 `yolov12-dmma-p2-efficient.yaml` 的训练结果 (mAP50=0.809, Recall=0.723) 进一步优化，目标提升至 mAP50>0.85, Recall>0.80。

## 改进方案

### 1. 模型架构增强
- **新增 `yolov12-dmma-p2-advanced.yaml`**：
  - 所有 Head 层启用 SW-MSA (`shift=True`) 增强空间覆盖
  - 多尺度窗口配置：P2/P3 使用 `[2,4]`，P4 使用 `[4,8]`，P5 使用 `[8,16]`
  - 新增 SPPF 层于 P5 backbone 后，增强多尺度特征融合
  - P2 层 num_heads 增至 8，mlp_ratio=2.0，专注微小目标
  - P5 输出使用 C2fDMMA 替代 C3k2，保持 DMMA 注意力一致性

### 2. 空间注意力增强
- **新增 `DMMAChannelSpatialAttention` 类** (`transformer.py`)：
  - 在 ECA 通道注意力基础上，增加轻量空间注意力分支
  - 空间分支：Avg/Max 池化 → 7×7 Conv → Sigmoid
  - 有效抑制背景噪声，提升精度

### 3. 训练策略优化
- **新增 `train_dmma_p2_advanced.py`**：
  - AdamW 优化器 + weight_decay=0.05（Transformer 友好）
  - 渐进式分辨率：640(150e) → 800(100e) → 960(50e)
  - 强化增广：copy_paste=0.5, mixup=0.2, degrees=15°, flipud=0.5
  - 损失权重：box=10.0, cls=0.3（小目标优化）
  - NMS 优化：iou=0.5, max_det=500

## 代码改动汇总

| 文件 | 类型 | 改动 |
|------|------|------|
| `ultralytics/cfg/models/v12/yolov12-dmma-p2-advanced.yaml` | 新增 | 增强版模型配置 |
| `train_dmma_p2_advanced.py` | 新增 | 多阶段训练脚本 |
| `ultralytics/nn/modules/transformer.py` | 修改 | 添加 DMMAChannelSpatialAttention |
| `TRAIN_ON_LINUX.md` | 修改 | 完整训练指南 |

## 预期收益

| 指标 | 基线 | 目标 | 改进来源 |
|------|------|------|----------|
| mAP50 | 0.809 | 0.85+ | SPPF + MS-DMMA + SW-MSA |
| mAP50-95 | 0.331 | 0.40+ | 渐进分辨率 + SPPF |
| Recall | 0.723 | 0.80+ | copy_paste + box 权重 |
| Precision | 0.818 | 0.85+ | 空间注意力 + 增强 ECA |

## 使用方法

### Python 脚本
```bash
# 阶段1 (640)
python train_dmma_p2_advanced.py --phase 1

# 阶段2 (800)
python train_dmma_p2_advanced.py --phase 2

# 验证
python train_dmma_p2_advanced.py --phase val
```

### CLI 命令
详见 `TRAIN_ON_LINUX.md`

## 硬件要求
- RTX 4090 (24GB VRAM)
- Phase 1: batch=16, ~16GB
- Phase 2: batch=10, ~18GB
- Phase 3: batch=6, ~20GB

---

# DASA + SAQK-Mask 核心创新实现（2025-12-29 下午）

## 背景（基于 report.docx 分析）

用户报告提出了两个核心创新点来解决小目标检测中注意力机制的局限性：

1. **DASA (Dynamic Area Self-Attention, 动态区域自注意力)**：
   - 问题：YOLOv12 的固定窗口注意力在密集区域"特征混叠"，在稀疏区域"计算冗余"
   - 方案：先估计区域密度，再动态分配窗口大小
     - 高密度区 (D > τh) → 大窗口（例如 16×16）
     - 中密度区 (τl ≤ D ≤ τh) → 中窗口（例如 8×8）
     - 低密度区 (D < τl) → 小窗口（例如 4×4）

2. **SAQK-Mask (Scale-Aware Q-K Dynamic Mask, 尺度感知 Q-K 动态掩码)**：
   - 问题：小目标 Query 信号弱，在与背景/大目标 Key 竞争时 softmax 权重趋近 0
   - 方案：用前景概率 + 尺度先验控制 Q-K 交互权限
     - 小目标 Query → 更大的 Key 可见范围
     - 背景 Query → 只看局部 Key

## 实现方案

### 1. DensityAwareGate（密度感知门控）
`ultralytics/nn/modules/transformer.py`:

```python
class DensityAwareGate(nn.Module):
    """密度感知门控，用于 DASA 动态窗口选择"""
    def __init__(self, channels, num_branches=2):
        # 轻量密度估计网络：1x1 Conv + Sigmoid
        self.density_conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid()
        )
        # 门控 MLP：密度 → 分支权重
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_branches)
        )
```

### 2. MSDMMALayer 升级
`MSDMMALayer` 现在支持两种门控模式：
- `use_density_gate=True`：DASA 风格，密度驱动动态门控（默认）
- `use_density_gate=False`：原始静态可学习门控

### 3. SAQKMask（尺度感知 Q-K 掩码）
```python
class SAQKMask(nn.Module):
    """尺度感知 Q-K 动态掩码，增强小目标注意力"""
    def forward(self, x):
        # 1. 估计前景概率
        fg_prob = self.fg_conv(x)
        # 2. 尺度感知强度调制
        intensity = fg_prob * self.intensity_scale
        # 3. 生成注意力增强因子 [0.5, 2.0]
        mask = torch.sigmoid(self.boost_conv(intensity)) * 1.5 + 0.5
        return mask, fg_prob
```

### 4. DASALayer（完整 DASA 层）
三分支密度自适应注意力：
- `branch_small`：低密度区，小窗口
- `branch_medium`：中密度区，中窗口
- `branch_large`：高密度区，大窗口

### 5. 新模块封装
- `C2fDASA`：使用 DASALayer 的 C2f 块
- `C2fDMMA_SAQK`：DMMA + SAQK-Mask 联合模块

## 新增模型配置

| 配置文件 | 说明 | 适用场景 |
|----------|------|----------|
| `yolov12-dmma-p2-ultimate.yaml` | 完整 DASA + SAQK-Mask | 推荐：最高精度 |
| `yolov12-dmma-p2-dasa.yaml` | 仅 DASA（消融实验） | 验证 DASA 单独贡献 |
| `yolov12-dmma-p2-advanced.yaml` | MS-DMMA + SPPF | 中等复杂度 |

## 消融实验矩阵

按建议的实验顺序：

| # | 配置 | 变量 | 验证目标 |
|---|------|------|----------|
| 1 | `yolov12-dmma-p2-efficient.yaml` | 基线 | 复现 0.809/0.331/0.723 |
| 2 | `yolov12-dmma-p2-advanced.yaml` | +SPPF +密度门控 | MS-DMMA 提升 |
| 3 | `yolov12-dmma-p2-dasa.yaml` | +DASA 三分支 | 动态窗口贡献 |
| 4 | `yolov12-dmma-p2-ultimate.yaml` | +SAQK-Mask | 完整链路 |

## 代码改动汇总

| 文件 | 新增/修改 | 内容 |
|------|-----------|------|
| `transformer.py` | 新增 | `DensityAwareGate`, `SAQKMask`, `DASALayer` |
| `transformer.py` | 修改 | `MSDMMALayer` 支持 `use_density_gate` |
| `block.py` | 新增 | `C2fDASA`, `C2fDMMA_SAQK` |
| `tasks.py` | 修改 | 导入并注册新模块 |
| `yolov12-dmma-p2-ultimate.yaml` | 新增 | 完整 DASA+SAQK 配置 |
| `yolov12-dmma-p2-dasa.yaml` | 新增 | DASA 消融配置 |

## 预期收益

基于 report.docx 的理论分析：

| 改进项 | 预期提升 | 机制 |
|--------|----------|------|
| DASA 动态窗口 | Recall +3-5% | 高密度区更大感受野，低密度区更精准 |
| SAQK-Mask | mAP50-95 +2-4% | 小目标 Query 获得更大 Key 可见范围 |
| 联合效果 | 综合 +5-8% | 注意力资源自适应分配 + 背景竞争抑制 |

## 使用方法

```bash
# Ultimate 版本（推荐）
python train_dmma_p2_advanced.py --phase 1

# 或直接修改训练脚本中的 MODEL_CONFIG
MODEL_CONFIG = "ultralytics/cfg/models/v12/yolov12-dmma-p2-ultimate.yaml"
```

---

# Bug 修复（2025-12-29 下午 - 用户代码审查后）

根据用户的详细代码审查，修复了以下关键问题：

## 1. ImportError 修复 ✅
**问题**: `ultralytics/nn/modules/__init__.py` 缺少 `C2fDASA` 和 `C2fDMMA_SAQK` 导出

**修复**: 在 `__init__.py` 的 import 和 `__all__` 中添加了这两个模块

## 2. SAQK scale_factor 方向修复 ✅
**问题**: 原代码中 `intensity = fg_prob * scale_factor`，scale_factor 越小反而增益越小（与设计意图相反）

**修复**: 
- 改为 `intensity = fg_prob * inverse_scale * 10.0`
- `inverse_scale = 1 / scale_factor`，确保浅层（P2=4）获得更大增益
- YAML 中 scale_factor 现在使用 stride 值：P2=4, P3=8, P4=16

## 3. mask 范围收窄 ✅
**问题**: 原范围 [0.5, 2.0] 过大，多层叠加可能导致激活尺度漂移

**修复**: 
- 使用残差式调制：`mask = 1.0 + alpha * tanh(boost)`
- `alpha` 从 0 开始训练（warmup 友好），最大值限制为 0.2
- 实际范围约 [0.8, 1.2]

## 4. BatchNorm 替换为 GroupNorm/无BN ✅
**问题**: 高分辨率训练阶段 batch 很小（4-6），BatchNorm 统计不稳定

**修复**:
- `DensityAwareGate`: 移除 BN，使用 bias=True 的 Conv
- `SAQKMask`: 替换为 GroupNorm
- `DASALayer.density_net`: 移除 BN，使用 bias=True 的 Conv

## 5. DASALayer mask 归一化 ✅
**问题**: 原代码 `mask_medium = 1 - mask_low - mask_high` 不保证严格和为 1

**修复**: 使用 softmax 归一化
```python
logits = torch.cat([logit_low, logit_medium, logit_high], dim=1)
weights = torch.softmax(logits, dim=1)  # 严格和为 1
```

## 6. YAML 注释乱码清理 ✅
**问题**: 火箭等 emoji 在某些终端显示为乱码

**修复**: 所有 YAML 文件改为纯 ASCII 注释

## 修复后的文件清单

| 文件 | 修复内容 |
|------|----------|
| `ultralytics/nn/modules/__init__.py` | +导出 C2fDASA, C2fDMMA_SAQK |
| `ultralytics/nn/modules/transformer.py` | SAQKMask/DensityAwareGate/DASALayer 修复 |
| `yolov12-dmma-p2-ultimate.yaml` | scale_factor 修正 + 注释清理 |
| `yolov12-dmma-p2-dasa.yaml` | 注释清理 |
| `yolov12-dmma-p2-advanced.yaml` | 注释清理 |
