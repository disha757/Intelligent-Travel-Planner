"""OCR preprocessing and registration-number normalization."""

import re
from typing import Any

import cv2
import easyocr


class PlateReader:
    def __init__(self):
        self.reader = easyocr.Reader(["en"], gpu=False)

    def read(self, plate_image: Any) -> str:
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.bilateralFilter(resized, 7, 50, 50)
        thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        raw_text = " ".join(self.reader.readtext(thresholded, detail=0))
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        return cleaned