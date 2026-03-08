# YOLOv12 Latency Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a minimal FastAPI server and Python client to measure end-to-end and server-step latency for YOLOv12n detection.

**Architecture:** Create `server_api.py` with a FastAPI app that loads YOLOv12n once and exposes `/detect` to return detections plus step timings. Create `client_benchmark.py` that uploads a local image, measures client-side timing, and prints combined JSON.

**Tech Stack:** Python, FastAPI, uvicorn, ultralytics YOLO, requests, Pillow, pytest.

---

### Task 1: Server API tests

**Files:**
- Create: `tests/test_latency_api.py`
- Test: `tests/test_latency_api.py`

**Step 1: Write the failing test**

```python
import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

class FakeBoxes:
    def __init__(self):
        self.xyxy = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=float)
        self.conf = np.array([0.9], dtype=float)
        self.cls = np.array([0], dtype=float)

class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes()
        self.names = {0: "person"}

class FakeModel:
    def predict(self, source=None, imgsz=640, conf=0.25):
        return [FakeResult()]


def _make_test_image_bytes():
    image = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_detect_returns_timings_and_detections():
    from server_api import create_app

    app = create_app(model=FakeModel())
    client = TestClient(app)

    files = {"file": ("test.png", _make_test_image_bytes(), "image/png")}
    resp = client.post("/detect", files=files)

    assert resp.status_code == 200
    data = resp.json()

    assert "detections" in data
    assert "server_timings_ms" in data
    assert "server_total_ms" in data

    assert isinstance(data["server_total_ms"], (int, float))
    for key in ["read_file", "decode_image", "preprocess_infer", "postprocess", "serialize"]:
        assert key in data["server_timings_ms"]
        assert data["server_timings_ms"][key] >= 0

    assert len(data["detections"]) == 1
    det = data["detections"][0]
    assert det["class_id"] == 0
    assert det["class_name"] == "person"


def test_detect_rejects_bad_image():
    from server_api import create_app

    app = create_app(model=FakeModel())
    client = TestClient(app)

    files = {"file": ("bad.bin", b"not-an-image", "application/octet-stream")}
    resp = client.post("/detect", files=files)
    assert resp.status_code == 400
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_latency_api.py -v`
Expected: FAIL with `ModuleNotFoundError` for `server_api` (or missing `create_app`).

**Step 3: Write minimal implementation**

Create `server_api.py` with `create_app()` and `/detect` endpoint that matches the test contract.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_latency_api.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_latency_api.py server_api.py
git commit -m "feat: add FastAPI detection endpoint with timing"
```

---

### Task 2: Client benchmark tests

**Files:**
- Create: `tests/test_client_benchmark.py`
- Test: `tests/test_client_benchmark.py`

**Step 1: Write the failing test**

```python
import json
from pathlib import Path

class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

class FakeSession:
    def __init__(self, payload):
        self._payload = payload

    def post(self, url, files=None, params=None, timeout=None):
        return FakeResponse(self._payload)


def test_run_benchmark_combines_client_and_server(tmp_path):
    from client_benchmark import run_benchmark

    image_path = tmp_path / "test.jpg"
    image_path.write_bytes(b"fake-image-bytes")

    server_payload = {
        "detections": [],
        "server_timings_ms": {"read_file": 1, "decode_image": 2, "preprocess_infer": 3, "postprocess": 4, "serialize": 5},
        "server_total_ms": 15,
    }
    session = FakeSession(server_payload)

    result = run_benchmark(str(image_path), "http://localhost:8000/detect", 640, 0.25, session=session)

    assert "client_total_ms" in result
    assert "client_timings_ms" in result
    assert "detections" in result
    assert "server_timings_ms" in result
    assert result["server_total_ms"] == 15
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_client_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError` for `client_benchmark` (or missing `run_benchmark`).

**Step 3: Write minimal implementation**

Create `client_benchmark.py` with `run_benchmark()` and a CLI entrypoint.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_client_benchmark.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_client_benchmark.py client_benchmark.py
git commit -m "feat: add client benchmark for end-to-end latency"
```

---

### Task 3: Update dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Add dependencies**

Add these lines if missing:
```
fastapi==0.110.3
uvicorn==0.29.0
pillow==10.3.0
requests==2.32.3
```

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: add api and client dependencies"
```

---

### Task 4: Quick verification

**Step 1: Run all new tests**

Run: `pytest tests/test_latency_api.py tests/test_client_benchmark.py -v`
Expected: PASS

**Step 2: Manual smoke (optional)**

Run server:
```
python server_api.py
```
Run client:
```
python client_benchmark.py --image ultralytics/assets/bus.jpg
```
Expected: JSON output with `client_total_ms` and `server_timings_ms`.
```
