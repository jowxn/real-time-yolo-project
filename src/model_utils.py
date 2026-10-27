"""
Shared utility for loading and running any of our candidate YOLO models.
Keeping this in one place ensures identical code path across models —
important for a controlled comparison.
"""
from ultralytics import YOLO

# Candidate models for this project.
# Key = short name used everywhere in our code/output; value = weights filename.
CANDIDATE_MODELS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
    "yolo26n": "yolo26n.pt",
}

def load_model(model_key: str) -> YOLO:
    """Load a candidate model by its short key. Auto-downloads weights if needed."""
    if model_key not in CANDIDATE_MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from {list(CANDIDATE_MODELS.keys())}")
    weights_file = CANDIDATE_MODELS[model_key]
    print(f"Loading {model_key} ({weights_file})...")
    return YOLO(weights_file)