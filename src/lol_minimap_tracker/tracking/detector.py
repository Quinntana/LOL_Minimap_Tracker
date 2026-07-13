"""OpenCV detector preserving the original matching behavior."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from ..config import TrackerConfig
from ..domain.interfaces import Image
from ..domain.models import ChampionObservation, DetectionDiagnostics, DetectionFrame


@dataclass(frozen=True)
class _MatchDecision:
    champion_name: str | None
    score: float
    runner_up_score: float
    rejection: str | None = None


class OpenCvChampionDetector:
    MATCH_SIZE = 32
    MATCH_CROP_FRACTION = 0.60
    SSIM_WINDOW_SIZE = 7

    def __init__(self, config: TrackerConfig, logger: logging.Logger) -> None:
        self.logger = logger
        # MSS frames are converted from BGRA to BGR. Red wraps around both ends
        # of OpenCV's HSV hue range, so combine the two ranges explicitly.
        self.lower_red_hsv = np.array([0, 60, 80])
        self.upper_red_hsv = np.array([10, 255, 255])
        self.lower_red_hsv_wrap = np.array([170, 60, 80])
        self.upper_red_hsv_wrap = np.array([179, 255, 255])
        self.lower_white = np.array([200, 200, 200])
        self.upper_white = np.array([255, 255, 255])
        self.circle_radius_min = config.circle_radius_min
        self.circle_radius_max = config.circle_radius_max
        self.ssim_threshold = config.ssim_threshold
        self.ssim_margin = config.ssim_margin

    def red_mask(self, image: Image) -> Image:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_hue = cv2.inRange(hsv, self.lower_red_hsv, self.upper_red_hsv)
        high_hue = cv2.inRange(hsv, self.lower_red_hsv_wrap, self.upper_red_hsv_wrap)
        return cv2.bitwise_or(low_hue, high_hue)

    def detect_red_circles(self, image: Image) -> np.ndarray | None:
        mask = self.red_mask(image)
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
            if first.shape != second.shape:
                return 0.0
            first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
            second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
            window_size = OpenCvChampionDetector.SSIM_WINDOW_SIZE
            if min(first_gray.shape) < window_size:
                return 0.0

            first_float = first_gray.astype(np.float64, copy=False)
            second_float = second_gray.astype(np.float64, copy=False)

            def window_mean(image: Image) -> Image:
                return cast(
                    Image,
                    cv2.boxFilter(
                        image,
                        cv2.CV_64F,
                        (window_size, window_size),
                        normalize=True,
                        borderType=cv2.BORDER_REFLECT,
                    ),
                )

            first_mean = window_mean(first_float)
            second_mean = window_mean(second_float)
            sample_scale = window_size**2 / (window_size**2 - 1)
            first_variance = sample_scale * (
                window_mean(first_float * first_float) - first_mean * first_mean
            )
            second_variance = sample_scale * (
                window_mean(second_float * second_float) - second_mean * second_mean
            )
            covariance = sample_scale * (
                window_mean(first_float * second_float) - first_mean * second_mean
            )
            luminance_stabilizer = (0.01 * 255) ** 2
            contrast_stabilizer = (0.03 * 255) ** 2
            numerator = (2 * first_mean * second_mean + luminance_stabilizer) * (
                2 * covariance + contrast_stabilizer
            )
            denominator = (
                first_mean * first_mean + second_mean * second_mean + luminance_stabilizer
            ) * (first_variance + second_variance + contrast_stabilizer)
            score_map = numerator / denominator
            padding = (window_size - 1) // 2
            score = float(score_map[padding:-padding, padding:-padding].mean(dtype=np.float64))
            return score if np.isfinite(score) else 0.0
        except (ValueError, cv2.error):
            return 0.0

    @classmethod
    def _prepare_match_image(cls, image: Image) -> Image | None:
        if image.ndim != 3 or image.shape[0] < 7 or image.shape[1] < 7:
            return None
        side = min(image.shape[:2])
        crop_side = max(7, int(round(side * cls.MATCH_CROP_FRACTION)))
        crop_side = min(crop_side, side)
        top = (image.shape[0] - crop_side) // 2
        left = (image.shape[1] - crop_side) // 2
        crop = image[top : top + crop_side, left : left + crop_side]
        interpolation = cv2.INTER_AREA if crop_side >= cls.MATCH_SIZE else cv2.INTER_CUBIC
        return cast(
            Image,
            cv2.resize(crop, (cls.MATCH_SIZE, cls.MATCH_SIZE), interpolation=interpolation),
        )

    def score_matches(
        self, query: Image, portraits: Mapping[str, Image]
    ) -> list[tuple[str, float]]:
        matches: list[tuple[str, float]] = []
        prepared_query = self._prepare_match_image(query)
        if prepared_query is None:
            return matches
        for name, portrait in portraits.items():
            try:
                prepared_portrait = self._prepare_match_image(portrait)
                if prepared_portrait is None:
                    continue
                score = self.calculate_ssim(prepared_query, prepared_portrait)
                matches.append((name, score))
            except cv2.error as exc:
                self.logger.warning("Could not match %s: %s", name, exc)
        return sorted(matches, key=lambda match: (-match[1], match[0].casefold()))

    def _match(self, query: Image, portraits: Mapping[str, Image]) -> _MatchDecision:
        matches = self.score_matches(query, portraits)
        if not matches:
            return _MatchDecision(None, 0.0, 0.0, "below_threshold")
        best_name, best_score = matches[0]
        runner_up = matches[1][1] if len(matches) > 1 else 0.0
        if best_score <= self.ssim_threshold:
            return _MatchDecision(None, best_score, runner_up, "below_threshold")
        if len(matches) > 1 and best_score - runner_up < self.ssim_margin:
            return _MatchDecision(None, best_score, runner_up, "ambiguous")
        return _MatchDecision(best_name, best_score, runner_up)

    def find_best_match(
        self, query: Image, portraits: Mapping[str, Image]
    ) -> tuple[str | None, float]:
        decision = self._match(query, portraits)
        return decision.champion_name, decision.score

    def process(self, image: Image, portraits: Mapping[str, Image]) -> DetectionFrame:
        if image.size == 0:
            return DetectionFrame((), None, DetectionDiagnostics(portraits=len(portraits)))
        center = self.find_rectangle_center(image)
        circles = self.detect_red_circles(image)
        if circles is None:
            return DetectionFrame((), center, DetectionDiagnostics(portraits=len(portraits)))
        candidates: list[ChampionObservation] = []
        below_threshold = 0
        ambiguous = 0
        best_score: float | None = None
        best_margin = 0.0
        for x, y, radius in np.round(circles[0, :]).astype(int):
            y1, y2 = max(0, y - radius), min(image.shape[0], y + radius)
            x1, x2 = max(0, x - radius), min(image.shape[1], x + radius)
            if y1 >= y2 or x1 >= x2:
                continue
            region = image[y1:y2, x1:x2]
            if region.size == 0:
                continue
            decision = self._match(region, portraits)
            margin = decision.score - decision.runner_up_score
            if best_score is None or decision.score > best_score:
                best_score = decision.score
                best_margin = margin
            if decision.rejection == "below_threshold":
                below_threshold += 1
                continue
            if decision.rejection == "ambiguous":
                ambiguous += 1
                continue
            if decision.champion_name is not None:
                candidates.append(
                    ChampionObservation(decision.champion_name, int(x), int(y), decision.score)
                )

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
            portraits=len(portraits),
            circles=len(circles[0]),
            accepted=len(observations),
            below_threshold=below_threshold,
            ambiguous=ambiguous,
            duplicate=duplicate,
            best_score=best_score if best_score is not None else 0.0,
            best_margin=best_margin,
        )
        return DetectionFrame(tuple(observations), center, diagnostics)
