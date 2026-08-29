"""YOLO vehicle detection with configurable weights."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class VehicleDetector:

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.35,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Vehicle model not found at {model_path}. "
                "Download a YOLOv8 model or set "
                "VEHICLE_MODEL_PATH."
            )

        self.model = YOLO(
            str(model_path)
        )

        self.confidence_threshold = (
            confidence_threshold
        )

    def detect(
        self,
        frame: Any,
    ) -> list[Detection]:

        result = self.model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
        )[0]

        names = result.names

        detections: list[Detection] = []

        for box in result.boxes:

            label = names[
                int(box.cls[0])
            ]

            confidence = float(
                box.conf[0]
            )

            x1, y1, x2, y2 = (
                int(value)
                for value in box.xyxy[0]
            )

            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    box=(x1, y1, x2, y2),
                )
            )

        return detections