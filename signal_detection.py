"""Stable traffic signal state detection."""

from dataclasses import dataclass
from typing import Any

import cv2


@dataclass
class SignalState:
    state: str
    confidence: float


class TrafficSignalDetector:
    """
    Stable traffic signal detector calibrated for the supplied
    1280x720 traffic video.

    Approximate bulb positions:

        RED    -> x=758, y=38
        YELLOW -> x=758, y=52
        GREEN  -> x=758, y=66

    The detector uses:
    - HSV color detection
    - multi-frame smoothing
    - hysteresis to reduce signal flickering
    - confidence stabilization
    """

    def __init__(
        self,
        x: int = 758,
        red_y: int = 38,
        yellow_y: int = 52,
        green_y: int = 66,
        radius: int = 6,
    ):
        self.x = x

        self.red_y = red_y
        self.yellow_y = yellow_y
        self.green_y = green_y

        self.radius = radius

        # ---------------------------------------------
        # Temporal smoothing
        # ---------------------------------------------

        self.previous_state = "UNKNOWN"
        self.previous_confidence = 0.0

        # Exponential moving average for scores
        self.smoothed_scores = {
            "RED": 0.0,
            "YELLOW": 0.0,
            "GREEN": 0.0,
        }

        # Smoothing factor
        # Lower = more stable
        # Higher = faster response
        self.alpha = 0.30

        # Minimum score required
        self.min_score = 0.08

        # Minimum advantage required before switching
        # from one signal to another.
        self.switch_margin = 0.06

    # -------------------------------------------------
    # ROI
    # -------------------------------------------------

    def _get_roi(
        self,
        frame: Any,
        x: int,
        y: int,
    ):
        h, w = frame.shape[:2]

        x1 = max(
            0,
            x - self.radius,
        )

        y1 = max(
            0,
            y - self.radius,
        )

        x2 = min(
            w,
            x + self.radius + 1,
        )

        y2 = min(
            h,
            y + self.radius + 1,
        )

        return frame[
            y1:y2,
            x1:x2,
        ]

    # -------------------------------------------------
    # Calculate color percentage
    # -------------------------------------------------

    def _mask_score(
        self,
        roi,
        lower,
        upper,
    ) -> float:

        if roi is None or roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV,
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper,
        )

        total_pixels = max(
            1,
            roi.shape[0] * roi.shape[1],
        )

        return (
            cv2.countNonZero(mask)
            / total_pixels
        )

    # -------------------------------------------------
    # RED
    # -------------------------------------------------

    def _red_score(
        self,
        roi,
    ) -> float:

        if roi is None or roi.size == 0:
            return 0.0

        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV,
        )

        red1 = cv2.inRange(
            hsv,
            (0, 80, 80),
            (12, 255, 255),
        )

        red2 = cv2.inRange(
            hsv,
            (165, 80, 80),
            (180, 255, 255),
        )

        mask = cv2.bitwise_or(
            red1,
            red2,
        )

        total_pixels = max(
            1,
            roi.shape[0] * roi.shape[1],
        )

        return (
            cv2.countNonZero(mask)
            / total_pixels
        )

    # -------------------------------------------------
    # YELLOW
    # -------------------------------------------------

    def _yellow_score(
        self,
        roi,
    ) -> float:

        return self._mask_score(
            roi,
            (18, 80, 80),
            (38, 255, 255),
        )

    # -------------------------------------------------
    # GREEN
    # -------------------------------------------------

    def _green_score(
        self,
        roi,
    ) -> float:

        return self._mask_score(
            roi,
            (40, 70, 60),
            (95, 255, 255),
        )

    # -------------------------------------------------
    # Smooth scores
    # -------------------------------------------------

    def _smooth_scores(
        self,
        scores,
    ):
        for state in scores:
            self.smoothed_scores[state] = (
                self.alpha * scores[state]
                + (1.0 - self.alpha)
                * self.smoothed_scores[state]
            )

        return self.smoothed_scores.copy()

    # -------------------------------------------------
    # Calculate confidence
    # -------------------------------------------------

    def _calculate_confidence(
        self,
        best_score: float,
        second_score: float,
    ) -> float:

        if best_score <= 0:
            return 0.0

        # Base confidence
        base_confidence = min(
            best_score * 1.8,
            1.0,
        )

        # Separation from second-best color
        separation = max(
            0.0,
            best_score - second_score,
        )

        separation_bonus = min(
            separation * 1.5,
            0.20,
        )

        confidence = min(
            base_confidence
            + separation_bonus,
            1.0,
        )

        return confidence

    # -------------------------------------------------
    # Main detection
    # -------------------------------------------------

    def detect(
        self,
        frame: Any,
    ) -> SignalState:

        # ---------------------------------------------
        # Extract bulb ROIs
        # ---------------------------------------------

        red_roi = self._get_roi(
            frame,
            self.x,
            self.red_y,
        )

        yellow_roi = self._get_roi(
            frame,
            self.x,
            self.yellow_y,
        )

        green_roi = self._get_roi(
            frame,
            self.x,
            self.green_y,
        )

        if (
            red_roi.size == 0
            or yellow_roi.size == 0
            or green_roi.size == 0
        ):
            return SignalState(
                "UNKNOWN",
                0.0,
            )

        # ---------------------------------------------
        # Raw color scores
        # ---------------------------------------------

        raw_scores = {
            "RED": self._red_score(
                red_roi
            ),
            "YELLOW": self._yellow_score(
                yellow_roi
            ),
            "GREEN": self._green_score(
                green_roi
            ),
        }

        # ---------------------------------------------
        # Temporal smoothing
        # ---------------------------------------------

        scores = self._smooth_scores(
            raw_scores
        )

        # ---------------------------------------------
        # Sort signals by score
        # ---------------------------------------------

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        state = ranked[0][0]
        best_score = ranked[0][1]
        second_score = ranked[1][1]

        # ---------------------------------------------
        # Very weak color
        # ---------------------------------------------

        if best_score < self.min_score:

            # Keep previous valid signal if confidence
            # was already reasonably strong.
            if (
                self.previous_state != "UNKNOWN"
                and self.previous_confidence >= 0.30
            ):
                return SignalState(
                    self.previous_state,
                    max(
                        0.0,
                        self.previous_confidence * 0.97,
                    ),
                )

            self.previous_state = "UNKNOWN"
            self.previous_confidence = best_score

            return SignalState(
                "UNKNOWN",
                best_score,
            )

        # ---------------------------------------------
        # Confidence
        # ---------------------------------------------

        confidence = self._calculate_confidence(
            best_score,
            second_score,
        )

        # ---------------------------------------------
        # Hysteresis
        #
        # Prevent sudden signal switching when two
        # colors have almost identical scores.
        # ---------------------------------------------

        if (
            self.previous_state != "UNKNOWN"
            and state != self.previous_state
        ):

            previous_score = scores.get(
                self.previous_state,
                0.0,
            )

            score_difference = (
                best_score
                - previous_score
            )

            # If new signal isn't clearly stronger,
            # keep the previous state.
            if score_difference < self.switch_margin:

                confidence = max(
                    0.0,
                    self.previous_confidence * 0.98,
                )

                self.previous_confidence = confidence

                return SignalState(
                    self.previous_state,
                    confidence,
                )

        # ---------------------------------------------
        # Update previous state
        # ---------------------------------------------

        self.previous_state = state
        self.previous_confidence = confidence

        return SignalState(
            state,
            confidence,
        )