"""
Phase 4: Confirm all three candidate models load and run on the same test image.
"""
import time
from model_utils import load_model, CANDIDATE_MODELS

image_path = "data/raw/test_image.jpg"

for model_key in CANDIDATE_MODELS:
    model = load_model(model_key)

    start = time.time()
    results = model(image_path, conf=0.25, iou=0.45, verbose=False)
    elapsed = (time.time() - start) * 1000

    result = results[0]
    num_params = sum(p.numel() for p in model.model.parameters())

    print(f"\n--- {model_key} ---")
    print(f"Inference time: {elapsed:.2f} ms")
    print(f"Detections: {len(result.boxes)}")
    print(f"Parameters: {num_params:,}")

    result.save(filename=f"outputs/predictions/test_image_{model_key}.jpg")
    print(f"Saved: outputs/predictions/test_image_{model_key}.jpg")