"""
Phase 8: Input resolution experiment (Experiment 4).
For each resolution, measure accuracy (on COCO128) and speed (on test image).
"""
import time
import json
import numpy as np
from model_utils import load_model, CANDIDATE_MODELS

RESOLUTIONS = [320, 480, 640]
N_WARMUP = 5
N_TRIALS = 30
image_path = "data/raw/test_image.jpg"

results_summary = []

for model_key in CANDIDATE_MODELS:
    model = load_model(model_key)

    for res in RESOLUTIONS:
        # --- Accuracy at this resolution ---
        metrics = model.val(data="coco128.yaml", imgsz=res, conf=0.25, iou=0.45, verbose=False)

        # --- Speed at this resolution ---
        for _ in range(N_WARMUP):
            model(image_path, imgsz=res, conf=0.25, iou=0.45, verbose=False)

        latencies_ms = []
        for _ in range(N_TRIALS):
            start = time.perf_counter()
            model(image_path, imgsz=res, conf=0.25, iou=0.45, verbose=False)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        latencies_ms = np.array(latencies_ms)
        mean_latency = latencies_ms.mean()
        mean_fps = 1000.0 / mean_latency

        row = {
            "model": model_key,
            "resolution": res,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "mean_latency_ms": float(mean_latency),
            "mean_fps": float(mean_fps),
        }
        results_summary.append(row)
        print(f"{model_key} @ {res}: mAP50={row['mAP50']:.3f}, FPS={mean_fps:.2f}")

with open("outputs/metrics/resolution_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\nSaved resolution results to outputs/metrics/resolution_results.json")