# Real-Time Object Detection Using Lightweight YOLO Architectures

Controlled comparison of YOLOv8n, YOLO11n and YOLO26n on CPU-only hardware
(accuracy, speed, complexity, input resolution) plus a real-time webcam demo.

## Setup (Windows CMD)
python -m venv venv
venv\Scripts\activate.bat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics

## Run
python src\webcam_demo.py yolo11n
python src\phase5_accuracy_benchmark.py
python src\phase6_speed_benchmark.py
python src\phase7_complexity_benchmark.py
python src\phase8_resolution_experiment.py
python src\phase9_visualize.py

Measured results are in outputs\metrics (JSON) and outputs\graphs (PNG).
Authors: Jowan Joe Mathew, Paulsonsanio (Lovely Professional University)
