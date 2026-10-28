"""
Streamlit demo: real-time detection with lightweight YOLO models + benchmark results.
Run from the project root:
    python -m streamlit run app\app.py --server.fileWatcherType none
"""
import os, sys, json, glob, time, tempfile
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lightweight YOLO Demo", layout="wide")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(ROOT)                                   # weights/outputs resolve from the project root
sys.path.insert(0, os.path.join(ROOT, "src"))
MODEL_KEYS = ["yolov8n", "yolo11n", "yolo26n"]   # same keys as model_utils.CANDIDATE_MODELS

# Draw something immediately, before any heavy import (torch/ultralytics take a while).
st.title("Real-Time Object Detection with Lightweight YOLO")
st.caption("YOLOv8n vs YOLO11n vs YOLO26n  |  CPU-only evaluation")

# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
model_key = st.sidebar.selectbox("Model", MODEL_KEYS, index=1)
imgsz = st.sidebar.selectbox("Input resolution", [320, 480, 640], index=2)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("IoU threshold", 0.10, 0.90, 0.45, 0.05)
st.sidebar.caption("Defaults match the paper: conf 0.25, IoU 0.45, 640 px.")

tab_live, tab_bench = st.tabs(["Live Detection", "Benchmark Results"])


def load_json(name):
    p = os.path.join(ROOT, "outputs", "metrics", name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ---------------- Benchmark tab FIRST in code ----------------
# (the webcam loop below blocks the script, so anything after it would never render)
with tab_bench:
    st.info("Results: CPU-only (AMD Ryzen 7 7730U), COCO128 subset, conf 0.25, IoU 0.45, batch 1. "
            "COCO128 overlaps the models' training data, so accuracy is for relative comparison only.")
    acc, comp = load_json("accuracy_results.json"), load_json("complexity_results.json")
    spd, resj = load_json("speed_results.json"), load_json("resolution_results.json")

    if acc and comp and spd:
        df = pd.DataFrame(acc).merge(pd.DataFrame(comp), on="model").merge(pd.DataFrame(spd), on="model")
        cols = ["model", "precision", "recall", "mAP50", "mAP50_95", "parameters", "gflops",
                "model_size_mb", "mean_latency_ms", "std_latency_ms", "mean_fps"]
        st.subheader("Accuracy, complexity and speed at 640x640")
        st.dataframe(df[cols].round(4), width="stretch", hide_index=True)
    else:
        st.warning("Some JSON files are missing in outputs\\metrics. Re-run the benchmark scripts.")

    if resj:
        st.subheader("Effect of input resolution")
        st.dataframe(pd.DataFrame(resj).round(4), width="stretch", hide_index=True)

    pngs = sorted(glob.glob(os.path.join(ROOT, "outputs", "graphs", "*.png")))
    if pngs:
        st.subheader("Graphs")
        gcols = st.columns(2)
        for i, p in enumerate(pngs):
            gcols[i % 2].image(p, caption=os.path.basename(p).replace(".png", ""), width="stretch")
    else:
        st.warning("No graphs found in outputs\\graphs. Run src\\phase9_visualize.py first.")


# ---------------- Detection helpers (heavy imports happen lazily here) ----------------
@st.cache_resource(show_spinner="Loading model (first load can take 20-30 s)...")
def get_model(key):
    from model_utils import load_model
    return load_model(key)


def infer(model, frame, imgsz, conf, iou):
    r = model(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    return r.plot(), len(r.boxes), sum(r.speed.values())   # latency = pre + inference + post (ms)


def show_metrics(slot, name, fps, latency, n_obj):
    with slot.container():
        c = st.columns(4)
        c[0].metric("Model", name)
        c[1].metric("FPS", f"{fps:.1f}" if fps else "-")
        c[2].metric("Latency", f"{latency:.1f} ms")
        c[3].metric("Objects", n_obj)


def run_stream(cap, model, frame_slot, metric_slot):
    prev, fps = time.time(), None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img, n, lat = infer(model, frame, imgsz, conf, iou)
            now = time.time(); inst = 1.0 / max(now - prev, 1e-6); prev = now
            fps = inst if fps is None else 0.9 * fps + 0.1 * inst
            frame_slot.image(img, channels="BGR", width="stretch")
            show_metrics(metric_slot, model_key, fps, lat, n)
    finally:
        cap.release()


# ---------------- Live detection tab ----------------
with tab_live:
    source = st.radio("Input source", ["Webcam", "Image upload", "Video upload"], horizontal=True)
    frame_slot, metric_slot = st.empty(), st.empty()

    try:
        import cv2
        if source == "Image upload":
            up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if up is not None:
                model = get_model(model_key)
                frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
                img, n, lat = infer(model, frame, imgsz, conf, iou)
                frame_slot.image(img, channels="BGR", width="stretch")
                show_metrics(metric_slot, model_key, 1000.0 / lat if lat else None, lat, n)
                st.caption("Single-image latency includes first-call overhead; the Benchmark tab has warm-up-excluded timings.")

        elif source == "Video upload":
            up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
            if up is not None:
                model = get_model(model_key)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(up.read()); tmp.close()
                try:
                    run_stream(cv2.VideoCapture(tmp.name), model, frame_slot, metric_slot)
                finally:
                    os.remove(tmp.name)
                st.success("Video finished.")

        else:  # Webcam
            if st.toggle("Start webcam"):
                model = get_model(model_key)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open the webcam. Close Teams/Zoom/Camera app/other tabs using it, then retry.")
                else:
                    run_stream(cap, model, frame_slot, metric_slot)
            else:
                st.info("Switch on 'Start webcam'. Change model or resolution in the sidebar to compare live.")
    except Exception as e:      # show the real error on the page instead of a blank screen
        st.error("Detection could not start. Details below.")
        st.exception(e)
