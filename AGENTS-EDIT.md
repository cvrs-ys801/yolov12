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
