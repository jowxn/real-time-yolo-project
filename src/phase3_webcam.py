"""
Phase 3b: Live webcam inference using YOLO11n.
Press 'q' to quit.
"""
import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run inference on this frame
    results = model(frame, conf=0.25, iou=0.45, verbose=False)
    result = results[0]

    # Draw Ultralytics' built-in annotations (boxes + labels + confidence)
    annotated_frame = result.plot()

    # Compute end-to-end FPS (capture + inference + draw)
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # Overlay FPS and object count
    num_objects = len(result.boxes)
    cv2.putText(annotated_frame, f"Model: YOLO11n", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Objects: {num_objects}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO11n - Real-Time Detection (press 'q' to quit)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()