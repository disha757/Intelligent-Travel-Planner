"""Evidence frame annotation and persistence."""

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from tracking import Track
from violation_detection import ViolationEvent


def save_evidence(
    frame: Any,
    event: ViolationEvent,
    track: Track,
    plate_number: str,
    output_dir: Path,
) -> Path:
    annotated = frame.copy()
    x1, y1, x2, y2 = event.box
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 140), 3)
    label = (
        f"{event.violation_type.upper()} | vehicle {track.vehicle_id} | "
        f"{plate_number or 'PLATE PENDING'} | {event.confidence:.0%}"
    )
    cv2.rectangle(annotated, (x1, max(0, y1 - 34)), (x1 + 620, y1), (20, 28, 40), -1)
    cv2.putText(
        annotated,
        label,
        (x1 + 8, max(22, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{track.vehicle_id}.jpg"
    destination = output_dir / filename
    cv2.imwrite(str(destination), annotated)
    return destination