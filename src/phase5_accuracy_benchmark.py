"""
Phase 5: Accuracy benchmarking (Experiment 1) — precision, recall, mAP@0.5, mAP@0.5:0.95
for all candidate models on the same validation set.
"""
import json
from model_utils import load_model, CANDIDATE_MODELS

results_summary = []

for model_key in CANDIDATE_MODELS:
    model = load_model(model_key)

    # .val() auto-downloads coco128 the first time it's used
    metrics = model.val(data="coco128.yaml", conf=0.25, iou=0.45, verbose=False)

    row = {
        "model": model_key,
        "precision": float(metrics.box.mp),        # mean precision across classes
        "recall": float(metrics.box.mr),            # mean recall across classes
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    results_summary.append(row)
    print(f"\n--- {model_key} ---")
    for k, v in row.items():
        if k != "model":
            print(f"{k}: {v:.4f}")

# Save results for later use in graphs/tables
with open("outputs/metrics/accuracy_results.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\nSaved accuracy results to outputs/metrics/accuracy_results.json")