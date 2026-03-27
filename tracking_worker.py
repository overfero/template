"""
tracking_worker.py
==================
Worker script dijalankan oleh run_tracking_parallel.py via subprocess + taskset.
Hasilnya ditulis ke file JSON sementara, dibaca main process setelah selesai.

Usage (dijalankan otomatis oleh run_tracking_parallel.py):
    taskset -c 0-3 python3 tracking_worker.py --camera top --out /tmp/result_top.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ultralytics import YOLO
from ultralytics.models.yolo.detect.config import (
    CAMERA_TOP_SOURCE,
    CAMERA_BOTTOM_SOURCE,
    CAMERA_TOP_FROM_TOP,
    CAMERA_BOTTOM_FROM_TOP,
    LINE_TOP_CAMERA,
    LINE_BOTTOM_CAMERA,
)

MODEL_PATH = "ultralytics/checkpoint/yolo11n_cpu.onnx"
TRACKER    = "hybridsort.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["top", "bottom"], required=True)
    parser.add_argument("--out",    required=True, help="Path output JSON")
    args = parser.parse_args()

    is_top = args.camera == "top"
    source = CAMERA_TOP_SOURCE if is_top else CAMERA_BOTTOM_SOURCE
    camera_from_top = CAMERA_TOP_FROM_TOP if is_top else CAMERA_BOTTOM_FROM_TOP

    # Patch CAMERA_FROM_TOP & virtual line sesuai kamera ini
    import ultralytics.models.yolo.detect.config as _cfg
    import ultralytics.models.yolo.detect.predict as _pred
    _cfg.CAMERA_FROM_TOP = camera_from_top
    _pred.line = LINE_TOP_CAMERA if camera_from_top else LINE_BOTTOM_CAMERA

    model = YOLO(MODEL_PATH)

    all_frames: list[dict] = []
    frame_idx = 0
    t0 = time.perf_counter()

    for _ in model.track(
        source=source,
        half=False,
        device="cpu",
        persist=True,
        tracker=TRACKER,
        stream=True,
        verbose=False,
        save=True,
    ):
        frame_idx += 1
        predictor = model.predictor
        taken = list(getattr(predictor, "current_taken_result", []))
        all_frames.append({"frame": frame_idx, "taken": taken})

    elapsed = time.perf_counter() - t0
    fps = frame_idx / elapsed if elapsed > 0 else 0
    print(
        f"[{args.camera.upper()}] Selesai  "
        f"{frame_idx} frames  {elapsed:.1f}s  {fps:.1f} FPS",
        flush=True,
    )

    # Tulis hasil ke file JSON
    with open(args.out, "w") as f:
        json.dump(all_frames, f)


if __name__ == "__main__":
    main()
