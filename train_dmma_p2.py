#!/usr/bin/env python3
"""
YOLOv12-DMMA-P2 训练脚本
针对 MASATI 船舶检测数据集优化

作者: Antigravity
日期: 2025-12-18

使用方法:
    # 方法1: 直接运行Python脚本
    python train_dmma_p2.py
    
    # 方法2: 使用CLI命令 (参见文件末尾的注释)

硬件要求:
    - RTX 4090 (24GB VRAM)
    - batch_size=24 对于 P2 版本
    - batch_size=32 对于标准版本
"""

from ultralytics import YOLO

# ============================================================
# 配置参数
# ============================================================

# 模型配置选择:
# 1. yolov12-dmma-p2.yaml       - P2检测头版本 (推荐优先尝试)
# 2. yolov12-dmma-p2-sppf.yaml  - P2 + SPPF版本 (更强但更慢)
# 3. yolov12-dmma-ms.yaml       - 原多尺度版本 (基线对比)

MODEL_CONFIG = "ultralytics/cfg/models/v12/yolov12-dmma-p2.yaml"
DATA_CONFIG = "data/masati.yaml"

# 训练超参数
TRAIN_PARAMS = {
    # 基础参数
    "epochs": 300,                # 训练轮数
    "batch": 24,                  # P2版本建议24,显存更紧张
    "imgsz": 640,                 # 输入图像尺寸
    "device": 0,                  # GPU设备号
    
    # 学习率
    "lr0": 0.01,                  # 初始学习率
    "lrf": 0.01,                  # 最终学习率 = lr0 * lrf
    "momentum": 0.937,            # SGD动量
    "weight_decay": 0.0005,       # 权重衰减
    "warmup_epochs": 3.0,         # 预热轮数
    "warmup_momentum": 0.8,       # 预热动量
    
    # 数据增强 - 针对小目标优化
    "mosaic": 1.0,                # Mosaic增强概率
    "mixup": 0.15,                # MixUp增强 (小目标有效)
    "copy_paste": 0.3,            # 复制粘贴增强 (小目标有效)
    "scale": 0.5,                 # 尺度抖动范围
    "degrees": 10.0,              # 随机旋转角度
    "translate": 0.1,             # 随机平移
    "shear": 2.0,                 # 剪切变换
    "flipud": 0.5,                # 垂直翻转 (遥感图像适用)
    "fliplr": 0.5,                # 水平翻转
    "hsv_h": 0.015,               # 色调增强
    "hsv_s": 0.7,                 # 饱和度增强
    "hsv_v": 0.4,                 # 亮度增强
    "erasing": 0.4,               # 随机擦除
    
    # 训练策略
    "close_mosaic": 15,           # 最后N轮关闭Mosaic
    "patience": 50,               # 早停耐心值
    "save_period": 20,            # 每N轮保存检查点
    "amp": True,                  # 混合精度训练
    "workers": 8,                 # DataLoader工作进程数
    "cache": False,               # 是否缓存图像 (内存充足可True)
    
    # 输出设置
    "project": "runs/detect",
    "name": "dmma_p2_masati",
    "exist_ok": True,
    "verbose": True,
}


def train():
    """执行训练"""
    print("=" * 60)
    print("YOLOv12-DMMA-P2 训练开始")
    print(f"模型配置: {MODEL_CONFIG}")
    print(f"数据配置: {DATA_CONFIG}")
    print("=" * 60)
    
    # 加载模型
    model = YOLO(MODEL_CONFIG)
    
    # 开始训练
    results = model.train(
        data=DATA_CONFIG,
        **TRAIN_PARAMS
    )
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最佳模型: {results.save_dir}/weights/best.pt")
    print("=" * 60)
    
    return results


def validate(weights_path: str = None):
    """验证模型"""
    if weights_path is None:
        weights_path = f"runs/detect/{TRAIN_PARAMS['name']}/weights/best.pt"
    
    model = YOLO(weights_path)
    results = model.val(
        data=DATA_CONFIG,
        imgsz=640,
        device=0,
        verbose=True,
    )
    return results


def predict(weights_path: str = None, source: str = "MASATI/images/test"):
    """推理预测"""
    if weights_path is None:
        weights_path = f"runs/detect/{TRAIN_PARAMS['name']}/weights/best.pt"
    
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
    # 执行训练
    train()
    
    # 训练完成后可取消注释执行验证和预测
    # validate()
    # predict()


# ============================================================
# CLI 命令参考 (可替代Python脚本使用)
# ============================================================
"""
# 标准训练 - P2检测头版本 (推荐)
yolo detect train \
    model=ultralytics/cfg/models/v12/yolov12-dmma-p2.yaml \
    data=data/masati.yaml \
    epochs=300 \
    batch=24 \
    imgsz=640 \
    lr0=0.01 \
    device=0 \
    project=runs/detect \
    name=dmma_p2_masati \
    mosaic=1.0 \
    mixup=0.15 \
    copy_paste=0.3 \
    scale=0.5 \
    flipud=0.5 \
    patience=50 \
    close_mosaic=15 \
    amp=True

# P2+SPPF 终极版本 (更强精度,更慢速度)
yolo detect train \
    model=ultralytics/cfg/models/v12/yolov12-dmma-p2-sppf.yaml \
    data=data/masati.yaml \
    epochs=300 \
    batch=20 \
    imgsz=640 \
    lr0=0.008 \
    device=0 \
    project=runs/detect \
    name=dmma_p2_sppf_masati \
    mosaic=1.0 \
    mixup=0.15 \
    copy_paste=0.3 \
    patience=50 \
    amp=True

# 验证
yolo detect val \
    weights=runs/detect/dmma_p2_masati/weights/best.pt \
    data=data/masati.yaml \
    imgsz=640 \
    device=0

# 推理
yolo detect predict \
    weights=runs/detect/dmma_p2_masati/weights/best.pt \
    source=MASATI/images/test \
    conf=0.25 \
    device=0
"""
