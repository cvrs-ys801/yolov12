import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def test_run_benchmark_combines_client_and_server(tmpdir):
    from client_benchmark import run_benchmark

    image_path = tmpdir.join("test.jpg")
    image_path.write(b"fake-image-bytes")

    server_payload = {
        "detections": [],
        "server_timings_ms": {
            "read_file": 1,
            "decode_image": 2,
            "preprocess_infer": 3,
            "postprocess": 4,
            "serialize": 5,
        },
        "server_total_ms": 15,
    }
    session = FakeSession(server_payload)

    result = run_benchmark(
        str(image_path),
        "http://localhost:8000/detect",
        640,
        0.25,
        session=session,
    )

    assert "client_total_ms" in result
    assert "client_timings_ms" in result
    assert "detections" in result
    assert "server_timings_ms" in result
    assert result["server_total_ms"] == 15
