import io
import json
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


def ms(x):
  return x * 1000.0


def to_numpy(x):
  if hasattr(x, "detach"):
      x = x.detach()
  if hasattr(x, "cpu"):
      x = x.cpu()
  if hasattr(x, "numpy"):
      return x.numpy()
  return np.array(x)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("image", help="image path")
  parser.add_argument("--imgsz", type=int, default=640)
  parser.add_argument("--conf", type=float, default=0.25)
  args = parser.parse_args()

  read_start = time.perf_counter()
  image_bytes = Path(args.image).read_bytes()
  read_end = time.perf_counter()

  decode_start = time.perf_counter()
  img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
  img = np.array(img)
  decode_end = time.perf_counter()

  model = YOLO("yolov12n.pt")

  infer_start = time.perf_counter()
  results = model.predict(source=img, imgsz=args.imgsz, conf=args.conf)
  infer_end = time.perf_counter()

  post_start = time.perf_counter()
  detections = []
  for r in results:
      boxes = getattr(r, "boxes", None)
      if boxes is None:
          continue
      xyxy = to_numpy(boxes.xyxy)
      conf = to_numpy(boxes.conf)
      cls = to_numpy(boxes.cls)
      names = getattr(r, "names", {})
      for i in range(len(xyxy)):
          cid = int(cls[i])
          detections.append({
              "class_id": cid,
              "class_name": names.get(cid, str(cid)),
              "confidence": float(conf[i]),
              "bbox_xyxy": [float(v) for v in xyxy[i].tolist()],
          })
  post_end = time.perf_counter()

  ser_start = time.perf_counter()
  payload = {
      "detections": detections,
      "server_timings_ms": {
          "read_file": ms(read_end - read_start),
          "decode_image": ms(decode_end - decode_start),
          "preprocess_infer": ms(infer_end - infer_start),
          "postprocess": ms(post_end - post_start),
          "serialize": 0.0,
      },
  }
  payload["server_timings_ms"]["serialize"] = ms(time.perf_counter() - ser_start)
  payload["server_total_ms"] = sum(payload["server_timings_ms"].values())

  print(json.dumps(payload, ensure_ascii=False))
if __name__ == "__main__":
  main()