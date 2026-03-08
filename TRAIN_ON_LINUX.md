# Linux 训练指南 - YOLOv12-DMMA-P2-Advanced

**更新日期**: 2025-12-29

本指南面向 RTX 4090 (24GB VRAM) 用户，提供完整的多阶段训练命令。

---

## 概述

**新模型配置**: `yolov12-dmma-p2-advanced.yaml`

**主要改进**:
1. 多尺度 DMMA 窗口 ([4,8], [8,16]) 覆盖不同船舶尺寸
2. SPPF 层增强多尺度特征融合
3. 所有 Head 层启用 SW-MSA
4. AdamW 优化器 + 渐进式分辨率训练
5. 强化数据增广 (copy_paste=0.5)

**目标指标**:
| 指标 | 当前基线 | 目标 |
|------|----------|------|
| mAP50 | 0.809 | 0.85+ |
| mAP50-95 | 0.331 | 0.40+ |
| Recall | 0.723 | 0.80+ |
| Precision | 0.818 | 0.85+ |

---

## 方法 1: Python 脚本训练 (推荐)

### 阶段 1: 640分辨率预热
```bash
cd /usr/sangui/PythonProject/yolov12
CUDA_VISIBLE_DEVICES=0 python train_dmma_p2_advanced.py --phase 1
```

### 阶段 2: 800分辨率微调
```bash
CUDA_VISIBLE_DEVICES=0 python train_dmma_p2_advanced.py --phase 2
```

### 阶段 3: 960分辨率精调 (可选)
```bash
CUDA_VISIBLE_DEVICES=0 python train_dmma_p2_advanced.py --phase 3
```

### 单阶段训练 (300 epochs @ 800)
```bash
CUDA_VISIBLE_DEVICES=0 python train_dmma_p2_advanced.py --phase single
```

### 验证
```bash
CUDA_VISIBLE_DEVICES=0 python train_dmma_p2_advanced.py --phase val
```

---

## 方法 2: CLI 命令训练

### 阶段 1: 640分辨率预热 (150 epochs)
```bash
CUDA_VISIBLE_DEVICES=0 yolo detect train \
    model=ultralytics/cfg/models/v12/yolov12-dmma-p2-advanced.yaml \
    data=data/masati.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    optimizer=AdamW \
    lr0=0.001 \
    weight_decay=0.05 \
    warmup_epochs=5 \
    box=10.0 \
    cls=0.3 \
    dfl=1.5 \
    mosaic=1.0 \
    mixup=0.2 \
    copy_paste=0.5 \
    scale=0.9 \
    degrees=15.0 \
    translate=0.15 \
    flipud=0.5 \
    fliplr=0.5 \
    erasing=0.4 \
    close_mosaic=20 \
    patience=50 \
    amp=True \
    cos_lr=True \
    workers=8 \
    iou=0.5 \
    max_det=500 \
    project=runs/detect \
    name=dmma_p2_advanced_phase1
```

### 阶段 2: 800分辨率微调 (100 epochs)
```bash
CUDA_VISIBLE_DEVICES=0 yolo detect train \
    model=runs/detect/dmma_p2_advanced_phase1/weights/best.pt \
    data=data/masati.yaml \
    epochs=100 \
    batch=10 \
    imgsz=800 \
    optimizer=AdamW \
    lr0=0.0005 \
    weight_decay=0.05 \
    warmup_epochs=3 \
    box=10.0 \
    cls=0.3 \
    mosaic=0.8 \
    mixup=0.15 \
    copy_paste=0.4 \
    scale=0.7 \
    degrees=10.0 \
    flipud=0.5 \
    erasing=0.3 \
    close_mosaic=15 \
    patience=40 \
    amp=True \
    cos_lr=True \
    workers=8 \
    project=runs/detect \
    name=dmma_p2_advanced_phase2
```

### 阶段 3: 960分辨率精调 (50 epochs, 可选)
```bash
CUDA_VISIBLE_DEVICES=0 yolo detect train \
    model=runs/detect/dmma_p2_advanced_phase2/weights/best.pt \
    data=data/masati.yaml \
    epochs=50 \
    batch=6 \
    imgsz=960 \
    optimizer=AdamW \
    lr0=0.0002 \
    weight_decay=0.05 \
    warmup_epochs=2 \
    box=10.0 \
    cls=0.3 \
    mosaic=0.5 \
    mixup=0.1 \
    copy_paste=0.3 \
    scale=0.5 \
    degrees=5.0 \
    close_mosaic=10 \
    patience=30 \
    amp=True \
    workers=6 \
    project=runs/detect \
    name=dmma_p2_advanced_phase3
```

---

## 验证命令

### 标准验证
```bash
yolo detect val \
    model=runs/detect/dmma_p2_advanced_phase2/weights/best.pt \
    data=data/masati.yaml \
    imgsz=960 \
    device=0
```

### 低置信度验证 (查看极限召回率)
```bash
yolo detect val \
    model=runs/detect/dmma_p2_advanced_phase2/weights/best.pt \
    data=data/masati.yaml \
    imgsz=960 \
    conf=0.001 \
    device=0
```

### 推理预测
```bash
yolo detect predict \
    model=runs/detect/dmma_p2_advanced_phase2/weights/best.pt \
    source=MASATI/images/test \
    conf=0.25 \
    iou=0.45 \
    device=0 \
    save=True
```

---

## 训练策略说明

### 1. AdamW 优化器
相比 SGD, AdamW 对 Transformer 结构 (DMMA) 更友好：
- 自适应学习率减少梯度消失问题
- 解耦 weight decay 改善正则化效果

### 2. 渐进式分辨率
- **640**: 快速收敛，建立基础特征
- **800**: 提升中小目标检测
- **960**: 精调极小目标 (遥感场景)

### 3. Copy-Paste 增广
`copy_paste=0.5` 是本次改进的核心：
- 随机复制船舶实例粘贴到其他图像
- 强制模型学习密集目标场景
- 显著提升 Recall

### 4. 损失权重
```
box=10.0  # 小目标定位更重要
cls=0.3   # 单类场景降低分类权重
```

---

## 硬件资源估算 (RTX 4090)

| 阶段 | imgsz | batch | VRAM | 训练时长 |
|------|-------|-------|------|----------|
| 1 | 640 | 16 | ~16GB | ~4h |
| 2 | 800 | 10 | ~18GB | ~3h |
| 3 | 960 | 6 | ~20GB | ~2h |

---

## 对比实验建议

1. **基线对比**: `yolov12-dmma-p2-efficient.yaml` vs `yolov12-dmma-p2-advanced.yaml`
2. **消融实验**:
   - Copy-Paste: 0.3 vs 0.5 vs 0.7
   - 分辨率: 640 vs 800 vs 960
   - 优化器: SGD vs AdamW
3. **超参敏感性**: box 权重 (7.5, 10.0, 12.0)

---

## 历史版本

### Iteration 2 (旧版)
```bash
# 已弃用 - 使用上述新命令替代
CUDA_VISIBLE_DEVICES=2 yolo detect train \
  model=runs/detect/dmma_p2_recall_opt_stable/weights/best.pt \
  ...
```
