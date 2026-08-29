"""Main traffic violation detection pipeline."""

from pathlib import Path

import cv2

from config import PipelineConfig
from signal_detection import TrafficSignalDetector
from vehicle_detection import VehicleDetector
from helmet_detection import HelmetDetector
from tracking import Tracker
from violation_detection import evaluate_violations


def main() -> None:

    # =========================================================
    # CONFIG
    # =========================================================

    config = PipelineConfig()

    video_path = (
        Path(__file__).resolve().parent
        / "dataset"
        / "traffic.mp4"
    )

    if not video_path.exists():

        print(
            f"ERROR: Video not found: "
            f"{video_path}"
        )

        return

    # =========================================================
    # VEHICLE DETECTOR
    # =========================================================

    print("Loading vehicle model...")

    try:

        vehicle_detector = VehicleDetector(
            model_path=config.vehicle_model,
            confidence_threshold=0.35,
        )

    except Exception as exc:

        print(
            f"ERROR loading vehicle model: "
            f"{exc}"
        )

        return

    print(
        "Vehicle model loaded successfully."
    )

    # =========================================================
    # TRACKER
    # =========================================================

    tracker = Tracker(
        iou_threshold=0.25,
        max_missing_frames=15,
    )

    print("Tracker initialized.")

    # =========================================================
    # HELMET DETECTOR
    # =========================================================

    print("Loading helmet model...")

    try:

        helmet_detector = HelmetDetector(
            model_path=config.helmet_model,
            confidence_threshold=0.05,
        )

    except Exception as exc:

        print(
            f"ERROR loading helmet model: "
            f"{exc}"
        )

        return

    print(
        "Helmet model loaded successfully."
    )

    # =========================================================
    # TRAFFIC SIGNAL DETECTOR
    #
    # Calibrated for 1280x720 video
    #
    # RED    = 758, 38
    # YELLOW = 758, 52
    # GREEN  = 758, 66
    # =========================================================

    signal_detector = TrafficSignalDetector(
        x=758,
        red_y=38,
        yellow_y=52,
        green_y=66,
        radius=6,
    )

    print(
        "Traffic signal detector initialized "
        "(x=758, red=38, yellow=52, green=66)"
    )

    # =========================================================
    # VIDEO
    # =========================================================

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"ERROR: Cannot open video: "
            f"{video_path}"
        )

        return

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print()
    print(
        "=============================="
    )
    print(
        "VIDEO INFORMATION"
    )
    print(
        "=============================="
    )
    print(
        f"Width        : {width}"
    )
    print(
        f"Height       : {height}"
    )
    print(
        f"FPS          : {fps:.2f}"
    )
    print(
        f"Total frames : {total_frames}"
    )
    print(
        "=============================="
    )
    print()

    # =========================================================
    # TRACK POSITION MEMORY
    # =========================================================

    previous_positions: dict[
        int, int
    ] = {}

    # Prevent duplicate Red Light violations
    already_reported: set[int] = set()

    # Prevent duplicate No Helmet violations
    helmet_reported: set[int] = set()

    frame_number = 0
    total_violations = 0

    # =========================================================
    # MAIN LOOP
    # =========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # =====================================================
        # TRAFFIC SIGNAL
        # =====================================================

        signal = signal_detector.detect(
            frame
        )

        signal_state = signal.state
        signal_confidence = signal.confidence

        # =====================================================
        # VEHICLE DETECTION
        # =====================================================

        detections = (
            vehicle_detector.detect(
                frame
            )
        )

        # =====================================================
        # TRACKING
        # =====================================================

        tracks = tracker.update(
            detections
        )

        # =====================================================
        # HELMET DETECTION
        # =====================================================

        helmet_detections = (
            helmet_detector.detect(
                frame
            )
        )

        # =====================================================
        # VIOLATION DETECTION
        # =====================================================

        violations = evaluate_violations(
            tracks=tracks,
            previous_positions=previous_positions,
            signal_state=signal_state,
            config=config,
            helmet_detections=helmet_detections,
            already_reported=already_reported,
            helmet_reported=helmet_reported,
        )

        # =====================================================
        # HANDLE VIOLATIONS
        # =====================================================

        for violation in violations:

            total_violations += 1

            print(
                f"VIOLATION | "
                f"frame={frame_number} | "
                f"type={violation.violation_type} | "
                f"vehicle_id={violation.vehicle_id} | "
                f"confidence="
                f"{violation.confidence:.2f} | "
                f"signal={signal_state}"
            )

        # =====================================================
        # STOP LINE
        # =====================================================

        stop_line_y = (
            config.stop_line_y
        )

        cv2.line(
            frame,
            (0, stop_line_y),
            (width, stop_line_y),
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            "STOP LINE",
            (
                30,
                stop_line_y - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        # =====================================================
        # SIGNAL STATUS
        # =====================================================

        signal_text = (
            f"Signal: {signal_state} "
            f"{signal_confidence:.2f}"
        )

        cv2.putText(
            frame,
            signal_text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # =====================================================
        # SIGNAL BULB MARKERS
        # =====================================================

        cv2.circle(
            frame,
            (758, 38),
            6,
            (255, 255, 0),
            1,
        )

        cv2.circle(
            frame,
            (758, 52),
            6,
            (255, 255, 0),
            1,
        )

        cv2.circle(
            frame,
            (758, 66),
            6,
            (255, 255, 0),
            1,
        )

        # =====================================================
        # TRACKED VEHICLES
        # =====================================================

        for track in tracks:

            x1, y1, x2, y2 = (
                track.box
            )

            # Green = normal
            # Red = Red Light violation
            # Orange = No Helmet violation

            box_color = (
                0,
                255,
                0,
            )

            if (
                track.track_id
                in already_reported
            ):
                box_color = (
                    0,
                    0,
                    255,
                )

            if (
                track.track_id
                in helmet_reported
            ):
                box_color = (
                    0,
                    165,
                    255,
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            label = (
                f"{track.label} "
                f"ID:{track.track_id}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        20,
                        y1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
            )

            # Center point
            cx, cy = track.center

            cv2.circle(
                frame,
                (
                    int(cx),
                    int(cy),
                ),
                4,
                (255, 0, 0),
                -1,
            )

        # =====================================================
        # FRAME INFORMATION
        # =====================================================

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Vehicles: {len(tracks)}",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Violations: {total_violations}",
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # =====================================================
        # CONSOLE LOG
        # =====================================================

        if frame_number % 30 == 0:

            print(
                f"frame={frame_number} "
                f"signal={signal_state} "
                f"signal_confidence="
                f"{signal_confidence:.2f} "
                f"tracked_vehicles="
                f"{len(tracks)} "
                f"helmet_detections="
                f"{len(helmet_detections)}"
            )

        # =====================================================
        # DISPLAY
        # =====================================================

        cv2.imshow(
            "Intelligent Traffic "
            "Violation Detection",
            frame,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):
            break

    # =========================================================
    # CLEANUP
    # =========================================================

    cap.release()

    cv2.destroyAllWindows()

    print()
    print(
        "======================================"
    )
    print(
        "ANALYSIS COMPLETE"
    )
    print(
        "======================================"
    )
    print(
        f"Frames processed : "
        f"{frame_number}"
    )
    print(
        f"Total violations: "
        f"{total_violations}"
    )
    print(
        "======================================"
    )


if __name__ == "__main__":
    main()