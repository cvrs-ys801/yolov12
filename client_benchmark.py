import argparse
import json
import mimetypes
import time
from pathlib import Path

import requests


def _ms(delta):
    return delta * 1000.0


def run_benchmark(image_path, url, imgsz, conf, timeout=30, session=None):
    if session is None:
        session = requests.Session()

    start_total = time.perf_counter()

    read_start = time.perf_counter()
    image_bytes = Path(image_path).read_bytes()
    read_end = time.perf_counter()

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "application/octet-stream"

    files = {"file": (Path(image_path).name, image_bytes, mime_type)}
    params = {"imgsz": imgsz, "conf": conf}

    request_start = time.perf_counter()
    response = session.post(url, files=files, params=params, timeout=timeout)
    request_end = time.perf_counter()

    parse_start = time.perf_counter()
    payload = response.json()
    parse_end = time.perf_counter()

    if response.status_code != 200:
        raise RuntimeError("Server returned status {}".format(response.status_code))

    end_total = time.perf_counter()

    client_timings = {
        "read_file": _ms(read_end - read_start),
        "request": _ms(request_end - request_start),
        "parse_response": _ms(parse_end - parse_start),
    }

    payload["client_timings_ms"] = client_timings
    payload["client_total_ms"] = _ms(end_total - start_total)

    return payload


def main():
    parser = argparse.ArgumentParser(description="YOLOv12 latency benchmark client")
    parser.add_argument("--image", required=True, help="Path to local image")
    parser.add_argument("--url", default="http://localhost:8000/detect", help="Detection URL")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    args = parser.parse_args()

    result = run_benchmark(args.image, args.url, args.imgsz, args.conf, args.timeout)

    output = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()