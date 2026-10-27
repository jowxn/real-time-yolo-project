"""
Phase 6: Rigorous speed/latency benchmarking (Experiment 2).
Warm-up iterations excluded; mean and std reported over repeated trials.
"""
import time
import json
import numpy as np
from model_utils import load_model, CANDIDATE_MODELS

image_path = "data/raw/test_image.jpg"
N_WARMUP = 10
N_TRIALS = 50

results_summary = []

for model_key in CANDIDATE_MODELS:
    model = load_model(model_key)

    # Warm-up (discarded)
    for _ in range(N_WARMUP):
        model(image_path, conf=0.25, iou=0.45, verbose=False)

    # Measured trials
    latencies_ms = []
    for _ in range(N_TRIALS):
        start = time.perf_counter()
        model(image_path, conf=0.25, iou=0.45, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    latencies_ms = np.array(latencies_ms)
    mean_latency = latencies_ms.mean()
    std_latency = latencies_ms.std()
    mean_fps = 1000.0 / mean_latency

    row = {
        "model": model_key,
        "mean_latency_ms": float(mean_latency),
        "std_latency_ms": float(std_latency),
        "mean_fps": float(mean_fps),
        "n_trials": N_TRIALS,
        "n_warmup": N_WARMUP,
    }
    results_summary.append(row)

    print(f"\n--- {model_key} ---")
    print(f"Mean latency: {mean_latency:.2f} ms (± {std_latency:.2f} ms)")
    print(f"Mean FPS: {mean_fps:.2f}")

with open("outputs/metrics/speed_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\nSaved speed results to outputs/metrics/speed_results.json")