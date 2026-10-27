"""
Phase 7: Model complexity comparison (Experiment 3).
Parameters, GFLOPs, and model file size for all candidate models.
"""
import os
import json
from model_utils import load_model, CANDIDATE_MODELS

results_summary = []

for model_key, weights_file in CANDIDATE_MODELS.items():
    model = load_model(model_key)

    # Parameter count
    num_params = sum(p.numel() for p in model.model.parameters())

    # GFLOPs at 640x640 (Ultralytics computes this internally; access via model.info())
    # model.info() prints and also returns (layers, params, gradients, GFLOPs)
    info = model.info(detailed=False, verbose=True, imgsz=640)
    gflops = info[3] if info is not None else None

    # File size on disk
    file_size_mb = os.path.getsize(weights_file) / (1024 * 1024)

    row = {
        "model": model_key,
        "parameters": int(num_params),
        "gflops": float(gflops) if gflops is not None else None,
        "model_size_mb": round(file_size_mb, 2),
    }
    results_summary.append(row)

    print(f"\n--- {model_key} ---")
    print(f"Parameters: {num_params:,}")
    print(f"GFLOPs: {gflops}")
    print(f"Model size: {file_size_mb:.2f} MB")

with open("outputs/metrics/complexity_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\nSaved complexity results to outputs/metrics/complexity_results.json")