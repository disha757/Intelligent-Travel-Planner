"""Simple IoU-based vehicle tracking."""

from dataclasses import dataclass

from vehicle_detection import Detection


@dataclass
class Track:
    track_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    center: tuple[int, int]


def calculate_iou(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(0, ix2 - ix1)
    intersection_height = max(0, iy2 - iy1)

    intersection = (
        intersection_width * intersection_height
    )

    if intersection == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


class Tracker:
    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_missing_frames: int = 15,
    ):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames

        self.next_id = 1

        self.tracks: dict[int, Track] = {}

        self.missing_frames: dict[int, int] = {}

    def _make_track(
        self,
        detection: Detection,
    ) -> Track:

        x1, y1, x2, y2 = detection.box

        center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2),
        )

        track = Track(
            track_id=self.next_id,
            label=detection.label,
            confidence=detection.confidence,
            box=detection.box,
            center=center,
        )

        self.next_id += 1

        return track

    def update(
        self,
        detections: list[Detection],
    ) -> list[Track]:

        if not detections:

            for track_id in list(self.missing_frames):
                self.missing_frames[track_id] += 1

            self._remove_old_tracks()

            return list(self.tracks.values())

        updated_tracks: dict[int, Track] = {}

        used_track_ids: set[int] = set()

        # -------------------------------------------------
        # Match new detections with existing tracks
        # -------------------------------------------------

        for detection in detections:

            best_track_id = None
            best_iou = 0.0

            for track_id, old_track in self.tracks.items():

                if track_id in used_track_ids:
                    continue

                # Match only same object category.
                if old_track.label != detection.label:
                    continue

                iou = calculate_iou(
                    old_track.box,
                    detection.box,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            # -------------------------------------------------
            # Existing track found
            # -------------------------------------------------

            if (
                best_track_id is not None
                and best_iou >= self.iou_threshold
            ):

                x1, y1, x2, y2 = detection.box

                center = (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                )

                updated_tracks[best_track_id] = Track(
                    track_id=best_track_id,
                    label=detection.label,
                    confidence=detection.confidence,
                    box=detection.box,
                    center=center,
                )

                used_track_ids.add(best_track_id)

                self.missing_frames[best_track_id] = 0

            # -------------------------------------------------
            # New track
            # -------------------------------------------------

            else:

                new_track = self._make_track(
                    detection
                )

                updated_tracks[
                    new_track.track_id
                ] = new_track

                self.missing_frames[
                    new_track.track_id
                ] = 0

                used_track_ids.add(
                    new_track.track_id
                )

        # -------------------------------------------------
        # Existing tracks that were not matched
        # -------------------------------------------------

        for track_id, old_track in self.tracks.items():

            if track_id in used_track_ids:
                continue

            self.missing_frames[track_id] = (
                self.missing_frames.get(track_id, 0) + 1
            )

            if (
                self.missing_frames[track_id]
                <= self.max_missing_frames
            ):
                updated_tracks[track_id] = old_track

        self.tracks = updated_tracks

        self._remove_old_tracks()

        return list(self.tracks.values())

    def _remove_old_tracks(self) -> None:

        remove_ids = [
            track_id
            for track_id, missing in self.missing_frames.items()
            if missing > self.max_missing_frames
        ]

        for track_id in remove_ids:

            self.missing_frames.pop(
                track_id,
                None,
            )

            self.tracks.pop(
                track_id,
                None,
            )