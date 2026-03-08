import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
