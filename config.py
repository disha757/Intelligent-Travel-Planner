"""Configuration for the traffic analysis pipeline.

Keep camera geometry and model paths outside the processing code so each
intersection can have its own calibrated configuration.
"""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PipelineConfig:

    # --------------------------------------------------
    # Detection models
    # --------------------------------------------------
    vehicle_model: Path = BASE_DIR / "models" / "yolov8n.pt"

    plate_model: Path = BASE_DIR / "models" / "plate_model.pt"

    helmet_model: Path = BASE_DIR / "ai" / "models" / "helmet_model.pt"

    # --------------------------------------------------
    # Evidence folder
    # --------------------------------------------------
    evidence_dir: Path = (
        BASE_DIR / "uploads" / "evidence"
    )

    # --------------------------------------------------
    # Traffic configuration
    # --------------------------------------------------
    stop_line_y: int = int(
        os.getenv("STOP_LINE_Y", "420")
    )

    signal_x: int = int(
        os.getenv("SIGNAL_X", "755")
    )

    signal_y: int = int(
        os.getenv("SIGNAL_Y", "55")
    )

    allowed_direction: str = os.getenv(
        "ALLOWED_DIRECTION",
        "northbound",
    )

    # --------------------------------------------------
    # Backend API
    # --------------------------------------------------
    api_url: str = os.getenv(
        "TRAFFIC_API_URL",
        "http://localhost:5000/api",
    )


def load_config() -> PipelineConfig:
    config = PipelineConfig()

    config.evidence_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return config