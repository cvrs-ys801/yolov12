#!/usr/bin/env python3
"""Cross-platform trainer for YOLOv12-DMMA-P2-Efficient on MASATI.

Designed for development on Windows and training on Linux.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "ultralytics/cfg/models/v12/yolov12-dmma-p2-efficient.yaml"
DEFAULT_DATA = "data/masati.yaml"
DEFAULT_SOURCE = "MASATI/images/test"


def norm_path(value: str) -> str:
    """Normalize path separators for current OS while preserving relative paths."""
    return str(Path(value))


def parse_device(device: str | int | None) -> str | int:
    """Accept values like 0, '0', '0,1', 'cpu', or empty (auto)."""
    if device is None:
        return 0
    text = str(device).strip()
    if not text:
        return 0
    if text.isdigit():
        return int(text)
    return text


def base_params(device: str | int) -> dict[str, Any]:
    return {
        "epochs": 300,
        "batch": 16,
        "imgsz": 640,
        "device": device,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.05,
        "warmup_epochs": 3.0,
        "box": 10.0,
        "cls": 0.3,
        "dfl": 1.5,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.5,
        "scale": 0.9,
        "degrees": 15.0,
        "translate": 0.15,
        "flipud": 0.5,
        "fliplr": 0.5,
        "erasing": 0.4,
        "close_mosaic": 20,
        "patience": 50,
        "amp": True,
        "workers": 8,
        "project": "runs/detect",
        "name": "dmma_p2_efficient_masati",
        "exist_ok": True,
        "verbose": True,
    }


def train(model_cfg: str, data_cfg: str, device: str | int, epochs: int, batch: int, imgsz: int):
    from ultralytics import YOLO

    model = YOLO(norm_path(model_cfg))
    params = base_params(device)
    params.update({"epochs": epochs, "batch": batch, "imgsz": imgsz})
    return model.train(data=norm_path(data_cfg), **params)


def validate(weights: str, data_cfg: str, device: str | int, imgsz: int):
    from ultralytics import YOLO

    model = YOLO(norm_path(weights))
    return model.val(data=norm_path(data_cfg), imgsz=imgsz, device=device, verbose=True)


def predict(weights: str, source: str, device: str | int):
    from ultralytics import YOLO

    model = YOLO(norm_path(weights))
    return model.predict(
        source=norm_path(source),
        conf=0.25,
        iou=0.45,
        device=device,
        save=True,
        verbose=True,
    )


def default_best(weights_name: str) -> str:
    return str(ROOT / "runs" / "detect" / weights_name / "weights" / "best.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv12-DMMA-P2-Efficient training helper")
    parser.add_argument("mode", choices=["train", "val", "predict"], nargs="?", default="train")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model yaml or weights path")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Dataset yaml")
    parser.add_argument("--weights", default=None, help="Weights for val/predict")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Predict source")
    parser.add_argument("--device", default=os.getenv("YOLO_DEVICE", "0"), help="0 | 0,1 | cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="dmma_p2_efficient_masati")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = parse_device(args.device)

    if args.mode == "train":
        results = train(args.model, args.data, device, args.epochs, args.batch, args.imgsz)
        print(f"Training complete. Best: {results.save_dir}/weights/best.pt")
        return

    weights = args.weights or default_best(args.name)
    if not Path(weights).exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    if args.mode == "val":
        validate(weights, args.data, device, args.imgsz)
        print("Validation complete.")
        return

    predict(weights, args.source, device)
    print("Prediction complete.")


if __name__ == "__main__":
    main()
