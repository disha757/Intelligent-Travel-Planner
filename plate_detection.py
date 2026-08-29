"""License plate cropper using a separate configurable YOLO model."""

from pathlib import Path
from typing import Any

from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Plate model not found at {model_path}. Set PLATE_MODEL_PATH to a trained model."
            )
        self.model = YOLO(str(model_path))

    def crop_plate(self, vehicle_crop: Any) -> Any | None:
        result = self.model(vehicle_crop, verbose=False)[0]
        if not result.boxes:
            return None
        best_box = max(result.boxes, key=lambda box: float(box.conf[0]))
        x1, y1, x2, y2 = (int(value) for value in best_box.xyxy[0])
        return vehicle_crop[y1:y2, x1:x2]