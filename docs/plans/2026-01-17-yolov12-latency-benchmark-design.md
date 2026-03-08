# YOLOv12 Latency Benchmark Design (2026-01-17)

## Goal
Provide a minimal client/server workflow to measure:
- Client end-to-end latency (upload -> response) in milliseconds.
- Server-side step timings (read, decode, inference, postprocess, serialize).

## Scope
- Use the existing YOLOv12n model (yolov12n.pt).
- Run a FastAPI server for detection.
- Run a Python client script to upload one image and report timings.
- Return JSON only (no annotated image).

## Architecture
- server_api.py (FastAPI):
  - Loads YOLO model at startup.
  - Exposes POST /detect with multipart file upload.
  - Measures server-side steps with high-resolution timers.
  - Returns detections + server timings.

- client_benchmark.py (requests):
  - Reads local image file and uploads to /detect.
  - Measures client end-to-end time.
  - Prints combined JSON (client_total_ms + server payload).

## Data Flow
1) Client reads image from disk.
2) Client uploads file to /detect and starts timer.
3) Server reads file bytes and decodes image.
4) Server runs YOLOv12n prediction.
5) Server extracts detections (class_id, class_name, confidence, bbox_xyxy).
6) Server serializes JSON and returns response.
7) Client receives response and stops timer.

## API
POST /detect
- multipart/form-data: file=<image>
- query params (optional): imgsz=640, conf=0.25

Response JSON:
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.91,
      "bbox_xyxy": [x1, y1, x2, y2]
    }
  ],
  "server_timings_ms": {
    "read_file": 3.4,
    "decode_image": 2.1,
    "preprocess_infer": 18.7,
    "postprocess": 1.2,
    "serialize": 0.6
  },
  "server_total_ms": 26.7
}

Client output adds:
- client_total_ms

## Error Handling
- Return 400 if file is missing or image decode fails.
- Return 500 for unexpected errors with a short message.

## Testing
- Unit tests for JSON structure and timing keys.
- Basic integration test using a small sample image if available.

## Files to Add
- server_api.py
- client_benchmark.py
- tests/test_latency_api.py
- tests/test_client_benchmark.py

## Notes
- Use time.perf_counter() for timing.
- Model loaded once at process startup to avoid per-request load cost.