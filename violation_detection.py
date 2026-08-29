"""Traffic violation rule evaluation."""

from dataclasses import dataclass

from config import PipelineConfig
from tracking import Track
from helmet_detection import HelmetDetection


@dataclass
class ViolationEvent:
    violation_type: str
    vehicle_id: int
    confidence: float
    box: tuple[int, int, int, int]


def crossed_stop_line(
    previous_y: int | None,
    current_y: int,
    stop_line_y: int,
) -> bool:
    """
    Return True when the vehicle moves from above
    the stop line to the line or below it.
    """

    if previous_y is None:
        return False

    return (
        previous_y < stop_line_y
        and current_y >= stop_line_y
    )


def box_center(
    box: tuple[int, int, int, int],
) -> tuple[float, float]:

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def point_inside_box(
    point: tuple[float, float],
    box: tuple[int, int, int, int],
) -> bool:

    px, py = point

    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and y1 <= py <= y2
    )


def helmet_belongs_to_vehicle(
    helmet: HelmetDetection,
    track: Track,
) -> bool:
    """
    Check whether a helmet detection belongs
    to a tracked vehicle.
    """

    center = box_center(
        helmet.box
    )

    return point_inside_box(
        center,
        track.box,
    )


def evaluate_violations(
    tracks: list[Track],
    previous_positions: dict[int, int],
    signal_state: str,
    config: PipelineConfig,
    helmet_detections: list[HelmetDetection] | None = None,
    already_reported: set[int] | None = None,
    helmet_reported: set[int] | None = None,
) -> list[ViolationEvent]:

    violations: list[ViolationEvent] = []

    if already_reported is None:
        already_reported = set()

    if helmet_reported is None:
        helmet_reported = set()

    if helmet_detections is None:
        helmet_detections = []

    for track in tracks:

        vehicle_id = track.track_id

        current_y = int(
            track.center[1]
        )

        previous_y = previous_positions.get(
            vehicle_id
        )

        # =================================================
        # RED LIGHT VIOLATION
        # =================================================

        if (
            signal_state == "RED"
            and vehicle_id not in already_reported
        ):

            if crossed_stop_line(
                previous_y,
                current_y,
                config.stop_line_y,
            ):

                violations.append(
                    ViolationEvent(
                        violation_type="Red Light",
                        vehicle_id=vehicle_id,
                        confidence=0.90,
                        box=track.box,
                    )
                )

                already_reported.add(
                    vehicle_id
                )

        # =================================================
        # NO HELMET VIOLATION
        # =================================================

        if vehicle_id not in helmet_reported:

            for helmet in helmet_detections:

                label = (
                    helmet.label
                    .lower()
                    .strip()
                )

                if label != "without helmet":
                    continue

                if helmet_belongs_to_vehicle(
                    helmet,
                    track,
                ):

                    violations.append(
                        ViolationEvent(
                            violation_type="No Helmet",
                            vehicle_id=vehicle_id,
                            confidence=helmet.confidence,
                            box=track.box,
                        )
                    )

                    helmet_reported.add(
                        vehicle_id
                    )

                    break

        # =================================================
        # SAVE CURRENT POSITION
        # =================================================

        previous_positions[
            vehicle_id
        ] = current_y

    return violations