"""
Phase 3: Baseline static image inference using YOLO11n.
"""
from ultralytics import YOLO
import time

# 1. Load pretrained YOLO11n model (auto-downloads weights on first run)
model = YOLO("yolo11n.pt")

# 2. Run inference on a static image
image_path = "data/raw/test_image.jpg"

start = time.time()
results = model(image_path, conf=0.25, iou=0.45)
elapsed = time.time() - start

# 3. Inspect what came back
result = results[0]  # single image -> single result object
print(f"Inference time: {elapsed*1000:.2f} ms")
print(f"Number of detections: {len(result.boxes)}")

for box in result.boxes:
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()
    print(f"  {cls_name}: confidence={conf:.2f}, box={[round(v,1) for v in xyxy]}")

# 4. Save annotated image
result.save(filename="outputs/predictions/test_image_yolo11n.jpg")
print("Saved annotated image to outputs/predictions/test_image_yolo11n.jpg")