"""OpenCV detector preserving the original matching behavior."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from ..config import TrackerConfig
from ..domain.interfaces import Image
from ..domain.models import ChampionObservation, DetectionDiagnostics, DetectionFrame


class OpenCvChampionDetector:
    def __init__(self, config: TrackerConfig, logger: logging.Logger) -> None:
        self.logger = logger
        self.lower_red = np.array([100, 0, 0])
        self.upper_red = np.array([255, 100, 100])
        self.lower_white = np.array([200, 200, 200])
        self.upper_white = np.array([255, 255, 255])
        self.circle_radius_min = config.circle_radius_min
        self.circle_radius_max = config.circle_radius_max
        self.ssim_threshold = config.ssim_threshold
        self.ssim_margin = config.ssim_margin

    def detect_red_circles(self, image: Image) -> np.ndarray | None:
        mask = cv2.inRange(image, self.lower_red, self.upper_red)
        blurred = cv2.GaussianBlur(mask, (3, 3), 2)
        return cast(
            np.ndarray | None,
            cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=self.circle_radius_min,
                maxRadius=self.circle_radius_max,
            ),
        )

    def find_rectangle_center(self, image: Image) -> tuple[int, int] | None:
        mask = cv2.inRange(image, self.lower_white, self.upper_white)
        kernel: np.ndarray = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if 60 < width < 160 and 20 < height < 100:
                return x + width // 2, y + height // 2
        return None

    @staticmethod
    def calculate_ssim(first: Image, second: Image) -> float:
        try:
            first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
            second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
            score, _ = ssim(first_gray, second_gray, full=True)
            return float(score)
        except (ValueError, cv2.error):
            return 0.0

    def score_matches(
        self, query: Image, portraits: Mapping[str, Image]
    ) -> list[tuple[str, float]]:
        matches: list[tuple[str, float]] = []
        for name, portrait in portraits.items():
            try:
                resized = cv2.resize(portrait, (query.shape[1], query.shape[0]))
                score = self.calculate_ssim(query, resized)
                matches.append((name, score))
            except cv2.error as exc:
                self.logger.warning("Could not match %s: %s", name, exc)
        return sorted(matches, key=lambda match: (-match[1], match[0].casefold()))

    def find_best_match(
        self, query: Image, portraits: Mapping[str, Image]
    ) -> tuple[str | None, float]:
        matches = self.score_matches(query, portraits)
        if not matches or matches[0][1] <= self.ssim_threshold:
            return None, self.ssim_threshold
        best_name, best_score = matches[0]
        if len(matches) > 1 and best_score - matches[1][1] < self.ssim_margin:
            return None, best_score
        return best_name, best_score

    def process(self, image: Image, portraits: Mapping[str, Image]) -> DetectionFrame:
        if image.size == 0:
            return DetectionFrame((), None)
        center = self.find_rectangle_center(image)
        circles = self.detect_red_circles(image)
        if circles is None:
            return DetectionFrame((), center)
        candidates: list[ChampionObservation] = []
        below_threshold = 0
        ambiguous = 0
        for x, y, radius in np.round(circles[0, :]).astype(int):
            y1, y2 = max(0, y - radius), min(image.shape[0], y + radius)
            x1, x2 = max(0, x - radius), min(image.shape[1], x + radius)
            if y1 >= y2 or x1 >= x2:
                continue
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                continue
            matches = self.score_matches(region, portraits)
            if not matches or matches[0][1] <= self.ssim_threshold:
                below_threshold += 1
                continue
            champion, score = matches[0]
            if len(matches) > 1 and score - matches[1][1] < self.ssim_margin:
                ambiguous += 1
                continue
            candidates.append(ChampionObservation(champion, int(x), int(y), score))

        observations: list[ChampionObservation] = []
        assigned_champions: set[str] = set()
        duplicate = 0
        for candidate in sorted(
            candidates,
            key=lambda observation: (-observation.score, observation.champion_name.casefold()),
        ):
            if candidate.champion_name in assigned_champions:
                duplicate += 1
                continue
            assigned_champions.add(candidate.champion_name)
            observations.append(candidate)
        diagnostics = DetectionDiagnostics(
            circles=len(circles[0]),
            accepted=len(observations),
            below_threshold=below_threshold,
            ambiguous=ambiguous,
            duplicate=duplicate,
        )
        return DetectionFrame(tuple(observations), center, diagnostics)
