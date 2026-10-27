"""
Live webcam demo — works with any of the three candidate models.
Usage: python src\webcam_demo.py yolov8n
       python src\webcam_demo.py yolo11n
       python src\webcam_demo.py yolo26n
"""
import sys
import cv2
import time
from model_utils import load_model, CANDIDATE_MODELS

# Get model choice from command line, default to yolo11n if not specified
model_key = sys.argv[1] if len(sys.argv) > 1 else "yolo11n"

if model_key not in CANDIDATE_MODELS:
    print(f"Unknown model '{model_key}'. Choose from: {list(CANDIDATE_MODELS.keys())}")
    sys.exit(1)

model = load_model(model_key)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame, conf=0.25, iou=0.45, verbose=False)
    result = results[0]
    annotated_frame = result.plot()

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    num_objects = len(result.boxes)
    cv2.putText(annotated_frame, f"Model: {model_key}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow(f"{model_key} - Real-Time Detection (press 'q' to quit)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()