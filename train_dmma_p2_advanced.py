#!/usr/bin/env python3
"""
YOLOv12-DMMA-P2-Advanced 训练脚本
针对 MASATI 船舶检测数据集优化，提升 mAP50-95 和 Recall

作者: Antigravity
日期: 2025-12-29

关键改进:
    1. AdamW 优化器 - 更适合 Transformer 结构
    2. 渐进式分辨率训练 - 640 -> 800 -> 960
    3. 强化数据增广 - copy_paste=0.5, mixup=0.2
    4. 损失权重调优 - box=10.0, cls=0.3 (小目标优化)

使用方法:
    python train_dmma_p2_advanced.py

硬件要求:
    - RTX 4090 (24GB VRAM)
    - 推荐 batch_size: 640->16, 800->10, 960->6
"""

from ultralytics import YOLO
import os

# ============================================================
# 配置参数
# ============================================================

# 模型配置
MODEL_CONFIG = "ultralytics/cfg/models/v12/yolov12-dmma-p2-advanced.yaml"
DATA_CONFIG = "data/masati.yaml"

# 阶段1: 640分辨率预热训练
PHASE1_PARAMS = {
    # 基础参数
    "epochs": 150,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
    
    # 优化器 - AdamW 更适合 Transformer
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.05,  # AdamW 推荐较大 weight_decay
    "warmup_epochs": 5.0,
    "warmup_momentum": 0.8,
    
    # 损失权重 - 针对小目标优化
    "box": 10.0,   # 提高边框回归权重 (小目标定位更重要)
    "cls": 0.3,    # 降低分类权重 (单类场景)
    "dfl": 1.5,    # DFL 损失权重
    
    # 数据增强 - 强化小目标
    "mosaic": 1.0,
    "mixup": 0.2,
    "copy_paste": 0.5,      # 核心: 高概率复制粘贴增加目标数量
    "scale": 0.9,           # 更大尺度变化
    "degrees": 15.0,        # 旋转增强 (船舶方向多变)
    "translate": 0.15,
    "shear": 5.0,
    "flipud": 0.5,          # 垂直翻转 (遥感图像)
    "fliplr": 0.5,
    "hsv_h": 0.02,
    "hsv_s": 0.8,
    "hsv_v": 0.5,
    "erasing": 0.4,         # 随机擦除
    
    # 训练策略
    "close_mosaic": 20,     # 最后20轮关闭Mosaic
    "patience": 50,
    "save_period": 25,
    "amp": True,
    "workers": 8,
    "cache": False,
    "cos_lr": True,
    
    # NMS 参数
    "iou": 0.5,             # 降低 NMS IoU 阈值
    "max_det": 500,         # 增加最大检测数
    
    # 输出设置
    "project": "runs/detect",
    "name": "dmma_p2_advanced_phase1",
    "exist_ok": True,
    "verbose": True,
}

# 阶段2: 800分辨率微调
PHASE2_PARAMS = {
    "epochs": 100,
    "batch": 10,
    "imgsz": 800,
    "device": 0,
    
    "optimizer": "AdamW",
    "lr0": 0.0005,          # 微调使用更小学习率
    "lrf": 0.01,
    "weight_decay": 0.05,
    "warmup_epochs": 3.0,
    
    "box": 10.0,
    "cls": 0.3,
    "dfl": 1.5,
    
    "mosaic": 0.8,          # 降低 mosaic 概率
    "mixup": 0.15,
    "copy_paste": 0.4,
    "scale": 0.7,
    "degrees": 10.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "erasing": 0.3,
    
    "close_mosaic": 15,
    "patience": 40,
    "save_period": 20,
    "amp": True,
    "workers": 8,
    "cos_lr": True,
    
    "iou": 0.5,
    "max_det": 500,
    
    "project": "runs/detect",
    "name": "dmma_p2_advanced_phase2",
    "exist_ok": True,
    "verbose": True,
}

# 阶段3: 960分辨率精调 (可选)
PHASE3_PARAMS = {
    "epochs": 50,
    "batch": 6,
    "imgsz": 960,
    "device": 0,
    
    "optimizer": "AdamW",
    "lr0": 0.0002,          # 精调使用更小学习率
    "lrf": 0.01,
    "weight_decay": 0.05,
    "warmup_epochs": 2.0,
    
    "box": 10.0,
    "cls": 0.3,
    "dfl": 1.5,
    
    "mosaic": 0.5,          # 进一步降低
    "mixup": 0.1,
    "copy_paste": 0.3,
    "scale": 0.5,
    "degrees": 5.0,
    "flipud": 0.3,
    "fliplr": 0.5,
    "erasing": 0.2,
    
    "close_mosaic": 10,
    "patience": 30,
    "save_period": 10,
    "amp": True,
    "workers": 6,           # 减少 workers 以降低内存压力
    "cos_lr": True,
    
    "iou": 0.5,
    "max_det": 500,
    
    "project": "runs/detect",
    "name": "dmma_p2_advanced_phase3",
    "exist_ok": True,
    "verbose": True,
}


def train_phase1():
    """阶段1: 640分辨率预热训练"""
    print("=" * 60)
    print("Phase 1: 640 Resolution Training")
    print(f"Model: {MODEL_CONFIG}")
    print(f"Data: {DATA_CONFIG}")
    print("=" * 60)
    
    model = YOLO(MODEL_CONFIG)
    results = model.train(data=DATA_CONFIG, **PHASE1_PARAMS)
    
    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print("=" * 60)
    
    return results


def train_phase2(pretrained_weights=None):
    """阶段2: 800分辨率微调"""
    if pretrained_weights is None:
        pretrained_weights = f"runs/detect/{PHASE1_PARAMS['name']}/weights/best.pt"
    
    if not os.path.exists(pretrained_weights):
        print(f"Warning: {pretrained_weights} not found!")
        print("Please run Phase 1 first or specify correct weights path.")
        return None
    
    print("=" * 60)
    print("Phase 2: 800 Resolution Fine-tuning")
    print(f"Pretrained: {pretrained_weights}")
    print("=" * 60)
    
    model = YOLO(pretrained_weights)
    results = model.train(data=DATA_CONFIG, **PHASE2_PARAMS)
    
    print("\n" + "=" * 60)
    print("Phase 2 Complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print("=" * 60)
    
    return results


def train_phase3(pretrained_weights=None):
    """阶段3: 960分辨率精调 (可选)"""
    if pretrained_weights is None:
        pretrained_weights = f"runs/detect/{PHASE2_PARAMS['name']}/weights/best.pt"
    
    if not os.path.exists(pretrained_weights):
        print(f"Warning: {pretrained_weights} not found!")
        print("Please run Phase 2 first or specify correct weights path.")
        return None
    
    print("=" * 60)
    print("Phase 3: 960 Resolution Fine-tuning")
    print(f"Pretrained: {pretrained_weights}")
    print("=" * 60)
    
    model = YOLO(pretrained_weights)
    results = model.train(data=DATA_CONFIG, **PHASE3_PARAMS)
    
    print("\n" + "=" * 60)
    print("Phase 3 Complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")
    print("=" * 60)
    
    return results


def train_single_phase():
    """单阶段训练 (如果不想分阶段)"""
    params = {
        "epochs": 300,
        "batch": 12,
        "imgsz": 800,
        "device": 0,
        
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.05,
        "warmup_epochs": 5.0,
        
        "box": 10.0,
        "cls": 0.3,
        "dfl": 1.5,
        
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.5,
        "scale": 0.9,
        "degrees": 15.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "erasing": 0.4,
        
        "close_mosaic": 30,
        "patience": 60,
        "amp": True,
        "workers": 8,
        "cos_lr": True,
        
        "iou": 0.5,
        "max_det": 500,
        
        "project": "runs/detect",
        "name": "dmma_p2_advanced_single",
        "exist_ok": True,
        "verbose": True,
    }
    
    print("=" * 60)
    print("Single Phase Training (300 epochs @ 800)")
    print(f"Model: {MODEL_CONFIG}")
    print("=" * 60)
    
    model = YOLO(MODEL_CONFIG)
    results = model.train(data=DATA_CONFIG, **params)
    
    return results


def validate(weights_path=None):
    """验证模型"""
    if weights_path is None:
        # 尝试找到最新的训练结果
        for phase in ["phase3", "phase2", "phase1", "single"]:
            path = f"runs/detect/dmma_p2_advanced_{phase}/weights/best.pt"
            if os.path.exists(path):
                weights_path = path
                break
    
    if weights_path is None or not os.path.exists(weights_path):
        print("No trained weights found!")
        return None
    
    print(f"Validating: {weights_path}")
    model = YOLO(weights_path)
    
    # 标准验证
    results = model.val(
        data=DATA_CONFIG,
        imgsz=960,
        device=0,
        verbose=True,
    )
    
    # 低置信度验证 (查看极限召回率)
    print("\n--- Low Confidence Validation (conf=0.001) ---")
    results_low_conf = model.val(
        data=DATA_CONFIG,
        imgsz=960,
        device=0,
        conf=0.001,
        verbose=True,
    )
    
    return results


def predict(weights_path=None, source="MASATI/images/test"):
    """推理预测"""
    if weights_path is None:
        for phase in ["phase3", "phase2", "phase1", "single"]:
            path = f"runs/detect/dmma_p2_advanced_{phase}/weights/best.pt"
            if os.path.exists(path):
                weights_path = path
                break
    
    if weights_path is None or not os.path.exists(weights_path):
        print("No trained weights found!")
        return None
    
    print(f"Predicting with: {weights_path}")
    model = YOLO(weights_path)
    
    results = model.predict(
        source=source,
        conf=0.25,
        iou=0.45,
        device=0,
        save=True,
        verbose=True,
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOv12-DMMA-P2-Advanced Training")
    parser.add_argument("--phase", type=str, default="1", 
                        choices=["1", "2", "3", "single", "val", "predict"],
                        help="Training phase: 1, 2, 3, single, val, or predict")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights for phase 2/3")
    parser.add_argument("--source", type=str, default="MASATI/images/test",
                        help="Source for prediction")
    
    args = parser.parse_args()
    
    if args.phase == "1":
        train_phase1()
    elif args.phase == "2":
        train_phase2(args.weights)
    elif args.phase == "3":
        train_phase3(args.weights)
    elif args.phase == "single":
        train_single_phase()
    elif args.phase == "val":
        validate(args.weights)
    elif args.phase == "predict":
        predict(args.weights, args.source)


# ============================================================
# CLI 命令参考 (可替代 Python 脚本使用)
# ============================================================
"""
# 阶段1: 640分辨率预热
CUDA_VISIBLE_DEVICES=0 yolo detect train \\
    model=ultralytics/cfg/models/v12/yolov12-dmma-p2-advanced.yaml \\
    data=data/masati.yaml \\
    epochs=150 \\
    batch=16 \\
    imgsz=640 \\
    optimizer=AdamW \\
    lr0=0.001 \\
    weight_decay=0.05 \\
    box=10.0 \\
    cls=0.3 \\
    mosaic=1.0 \\
    mixup=0.2 \\
    copy_paste=0.5 \\
    scale=0.9 \\
    degrees=15.0 \\
    flipud=0.5 \\
    erasing=0.4 \\
    close_mosaic=20 \\
    patience=50 \\
    amp=True \\
    project=runs/detect \\
    name=dmma_p2_advanced_phase1

# 阶段2: 800分辨率微调
CUDA_VISIBLE_DEVICES=0 yolo detect train \\
    model=runs/detect/dmma_p2_advanced_phase1/weights/best.pt \\
    data=data/masati.yaml \\
    epochs=100 \\
    batch=10 \\
    imgsz=800 \\
    optimizer=AdamW \\
    lr0=0.0005 \\
    weight_decay=0.05 \\
    box=10.0 \\
    cls=0.3 \\
    mosaic=0.8 \\
    copy_paste=0.4 \\
    close_mosaic=15 \\
    patience=40 \\
    amp=True \\
    project=runs/detect \\
    name=dmma_p2_advanced_phase2

# 验证
yolo detect val \\
    model=runs/detect/dmma_p2_advanced_phase2/weights/best.pt \\
    data=data/masati.yaml \\
    imgsz=960 \\
    device=0

# 低置信度验证 (查看极限召回率)
yolo detect val \\
    model=runs/detect/dmma_p2_advanced_phase2/weights/best.pt \\
    data=data/masati.yaml \\
    imgsz=960 \\
    conf=0.001 \\
    device=0
"""
