"""
Phase 9: Generate all 8 graphs + 2 results tables from measured experiment data.
Reads only from outputs/metrics/*.json - no hardcoded numbers.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("outputs/graphs", exist_ok=True)

# --- Load all data ---
with open("outputs/metrics/accuracy_results.json") as f:
    accuracy = json.load(f)
with open("outputs/metrics/speed_results.json") as f:
    speed = json.load(f)
with open("outputs/metrics/complexity_results.json") as f:
    complexity = json.load(f)
with open("outputs/metrics/resolution_results.json") as f:
    resolution = json.load(f)

models = [row["model"] for row in accuracy]
colors = ["#4C72B0", "#55A868", "#C44E52"]  # consistent color per model across all graphs

def save_fig(fig, name):
    path = f"outputs/graphs/{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

# ---------- Graph 1: mAP comparison ----------
map50 = [row["mAP50"] for row in accuracy]
map5095 = [row["mAP50_95"] for row in accuracy]
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - width/2, map50, width, label="mAP@0.5", color="#4C72B0")
ax.bar(x + width/2, map5095, width, label="mAP@0.5:0.95", color="#C44E52")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("mAP")
ax.set_title("Detection Accuracy Comparison (COCO128, imgsz=640)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "graph1_map_comparison")

# ---------- Graph 2: FPS comparison ----------
fps = [row["mean_fps"] for row in speed]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(models, fps, color=colors)
ax.set_ylabel("Mean FPS")
ax.set_title("Inference Speed Comparison (CPU, imgsz=640)")
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, fps):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.1f}", ha="center")
save_fig(fig, "graph2_fps_comparison")

# ---------- Graph 3: Latency comparison (with std as error bars) ----------
latency = [row["mean_latency_ms"] for row in speed]
std = [row["std_latency_ms"] for row in speed]
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(models, latency, yerr=std, capsize=6, color=colors)
ax.set_ylabel("Mean Latency (ms)")
ax.set_title("Inference Latency Comparison (CPU, imgsz=640, mean ± std, n=50)")
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "graph3_latency_comparison")

# ---------- Graph 4: Model size comparison ----------
size_mb = [row["model_size_mb"] for row in complexity]
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(models, size_mb, color=colors)
ax.set_ylabel("Model Size (MB)")
ax.set_title("Model Size Comparison")
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "graph4_model_size_comparison")

# ---------- Graph 5: Parameters comparison ----------
params_m = [row["parameters"] / 1e6 for row in complexity]
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(models, params_m, color=colors)
ax.set_ylabel("Parameters (Millions)")
ax.set_title("Model Parameter Count Comparison")
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "graph5_parameters_comparison")

# ---------- Graph 6: Accuracy vs FPS trade-off ----------
fig, ax = plt.subplots(figsize=(7, 5))
for i, m in enumerate(models):
    ax.scatter(fps[i], map50[i], s=150, color=colors[i], label=m)
    ax.annotate(m, (fps[i], map50[i]), textcoords="offset points", xytext=(8, 5))
ax.set_xlabel("Mean FPS (CPU)")
ax.set_ylabel("mAP@0.5")
ax.set_title("Accuracy vs. Speed Trade-off (imgsz=640)")
ax.grid(alpha=0.3)
save_fig(fig, "graph6_accuracy_vs_fps_tradeoff")

# ---------- Graph 7: Resolution vs FPS ----------
fig, ax = plt.subplots(figsize=(7, 5))
for i, m in enumerate(models):
    rows = [r for r in resolution if r["model"] == m]
    rows.sort(key=lambda r: r["resolution"])
    res_vals = [r["resolution"] for r in rows]
    fps_vals = [r["mean_fps"] for r in rows]
    ax.plot(res_vals, fps_vals, marker="o", color=colors[i], label=m)
ax.set_xlabel("Input Resolution (px)")
ax.set_ylabel("Mean FPS")
ax.set_title("Effect of Input Resolution on FPS (CPU)")
ax.set_xticks([320, 480, 640])
ax.legend()
ax.grid(alpha=0.3)
save_fig(fig, "graph7_resolution_vs_fps")

# ---------- Graph 8: Resolution vs mAP ----------
fig, ax = plt.subplots(figsize=(7, 5))
for i, m in enumerate(models):
    rows = [r for r in resolution if r["model"] == m]
    rows.sort(key=lambda r: r["resolution"])
    res_vals = [r["resolution"] for r in rows]
    map_vals = [r["mAP50"] for r in rows]
    ax.plot(res_vals, map_vals, marker="o", color=colors[i], label=m)
ax.set_xlabel("Input Resolution (px)")
ax.set_ylabel("mAP@0.5")
ax.set_title("Effect of Input Resolution on Accuracy")
ax.set_xticks([320, 480, 640])
ax.legend()
ax.grid(alpha=0.3)
save_fig(fig, "graph8_resolution_vs_map")

# ---------- Results Table 1: Accuracy ----------
print("\n=== Table 1: Accuracy (COCO128, imgsz=640) ===")
print(f"{'Model':<10}{'Precision':>12}{'Recall':>10}{'mAP@0.5':>10}{'mAP@.5:.95':>12}")
for row in accuracy:
    print(f"{row['model']:<10}{row['precision']:>12.4f}{row['recall']:>10.4f}{row['mAP50']:>10.4f}{row['mAP50_95']:>12.4f}")

# ---------- Results Table 2: Speed + Complexity ----------
print("\n=== Table 2: Speed & Complexity (imgsz=640, CPU) ===")
print(f"{'Model':<10}{'Params(M)':>12}{'GFLOPs':>10}{'Size(MB)':>10}{'FPS':>8}{'Latency(ms)':>14}")
for m in models:
    c = next(r for r in complexity if r["model"] == m)
    s = next(r for r in speed if r["model"] == m)
    print(f"{m:<10}{c['parameters']/1e6:>12.2f}{c['gflops']:>10.2f}{c['model_size_mb']:>10.2f}{s['mean_fps']:>8.2f}{s['mean_latency_ms']:>14.2f}")

print("\nAll 8 graphs saved to outputs/graphs/")