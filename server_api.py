import io
import time
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image


def _ms(delta):
    return delta * 1000.0


def _decode_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image") from exc
    return np.array(image)


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.array(value)


def _extract_detections(results) -> List[Dict[str, Any]]:
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        xyxy = _to_numpy(boxes.xyxy)
        conf = _to_numpy(boxes.conf)
        cls = _to_numpy(boxes.cls)
        names = getattr(result, "names", {})
        for idx in range(len(xyxy)):
            class_id = int(cls[idx])
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": float(conf[idx]),
                    "bbox_xyxy": [float(v) for v in xyxy[idx].tolist()],
                }
            )
    return detections


def create_app(model=None) -> FastAPI:
    app = FastAPI()

    if model is not None:
        app.state.model = model
    else:
        app.state.model = None

        @app.on_event("startup")
        def _load_model():
            from ultralytics import YOLO

            app.state.model = YOLO("yolov12n.pt")

    @app.post("/detect")
    async def detect(
        file: UploadFile = File(...), imgsz: int = 640, conf: float = 0.25
    ):
        read_start = time.perf_counter()
        image_bytes = await file.read()
        read_end = time.perf_counter()

        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file")

        decode_start = time.perf_counter()
        try:
            image = _decode_image(image_bytes)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid image")
        decode_end = time.perf_counter()

        model = app.state.model
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        infer_start = time.perf_counter()
        results = model.predict(source=image, imgsz=imgsz, conf=conf)
        infer_end = time.perf_counter()

        post_start = time.perf_counter()
        detections = _extract_detections(results)
        post_end = time.perf_counter()

        serialize_start = time.perf_counter()
        timings = {
            "read_file": _ms(read_end - read_start),
            "decode_image": _ms(decode_end - decode_start),
            "preprocess_infer": _ms(infer_end - infer_start),
            "postprocess": _ms(post_end - post_start),
        }
        timings["serialize"] = _ms(time.perf_counter() - serialize_start)
        total = sum(timings.values())

        payload = {
            "detections": detections,
            "server_timings_ms": timings,
            "server_total_ms": total,
        }
        response = JSONResponse(content=payload)
        return response

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server_api:app", host="0.0.0.0", port=8000, reload=False)
