"""YOLO helmet detection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


@dataclass
class HelmetDetection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


class HelmetDetector:

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.05,
    ):

        if not model_path.exists():
            raise FileNotFoundError(
                f"Helmet model not found at {model_path}"
            )

        self.model = YOLO(
            str(model_path)
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        print(
            "Helmet model classes:",
            self.model.names,
        )

    def detect(
        self,
        frame: Any,
    ) -> list[HelmetDetection]:

        result = self.model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
        )[0]

        names = result.names

        detections: list[
            HelmetDetection
        ] = []

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            label = str(
                names[class_id]
            )

            x1, y1, x2, y2 = (
                int(value)
                for value in box.xyxy[0]
            )

            detections.append(
                HelmetDetection(
                    label=label,
                    confidence=confidence,
                    box=(
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                )
            )

        return detections

    def get_status(
        self,
        detections: list[HelmetDetection],
    ) -> str:

        for detection in detections:

            label = (
                detection.label
                .lower()
                .strip()
            )

            if label in {
                "with helmet",
                "helmet",
            }:
                return "WITH_HELMET"

            if label in {
                "without helmet",
                "no helmet",
                "without_helmet",
            }:
                return "WITHOUT_HELMET"

        return "UNKNOWN"