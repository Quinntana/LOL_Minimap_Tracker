from __future__ import annotations

import logging
from typing import Any

import numpy as np

from lol_minimap_tracker.config import TrackerConfig
from lol_minimap_tracker.tracking.detector import OpenCvChampionDetector


def test_detector_preserves_original_bgr_thresholds() -> None:
    detector = OpenCvChampionDetector(TrackerConfig(), logging.getLogger("test"))
    assert detector.lower_red.tolist() == [100, 0, 0]
    assert detector.upper_red.tolist() == [255, 100, 100]


def test_ssim_matches_identical_images() -> None:
    image = np.full((20, 20, 3), 127, dtype=np.uint8)
    assert OpenCvChampionDetector.calculate_ssim(image, image) == 1.0


def test_rectangle_center_on_synthetic_frame() -> None:
    detector = OpenCvChampionDetector(TrackerConfig(), logging.getLogger("test"))
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image[80:120, 50:150] = 255
    center = detector.find_rectangle_center(image)
    assert center is not None
    assert abs(center[0] - 100) <= 3
    assert abs(center[1] - 100) <= 3


def test_process_handles_no_circles_and_a_matched_crop(monkeypatch: Any) -> None:
    detector = OpenCvChampionDetector(TrackerConfig(), logging.getLogger("test"))
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    portraits = {"Aatrox": image.copy()}
    monkeypatch.setattr(detector, "find_rectangle_center", lambda _image: (20, 20))
    monkeypatch.setattr(detector, "detect_red_circles", lambda _image: None)
    assert not detector.process(image, portraits).observations

    circles = np.array([[[20, 20, 5]]], dtype=np.float32)
    monkeypatch.setattr(detector, "detect_red_circles", lambda _image: circles)
    monkeypatch.setattr(detector, "find_best_match", lambda _region, _portraits: ("Aatrox", 0.9))
    detection = detector.process(image, portraits)
    assert detection.observations[0].champion_name == "Aatrox"
    assert detection.camera_center == (20, 20)


def test_best_match_respects_threshold_and_handles_bad_images() -> None:
    detector = OpenCvChampionDetector(TrackerConfig(), logging.getLogger("test"))
    query = np.arange(20 * 20 * 3, dtype=np.uint8).reshape((20, 20, 3))
    name, score = detector.find_best_match(query, {"Exact": query.copy()})
    assert name == "Exact"
    assert score == 1.0

    strict = OpenCvChampionDetector(TrackerConfig(ssim_threshold=1.0), logging.getLogger("test"))
    assert strict.find_best_match(query, {"Exact": query.copy()})[0] is None
    assert detector.calculate_ssim(np.zeros((1, 1), dtype=np.uint8), query) == 0.0


def test_blank_frame_has_no_detected_circles() -> None:
    detector = OpenCvChampionDetector(TrackerConfig(), logging.getLogger("test"))
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect_red_circles(blank) is None
    assert not detector.process(np.array([], dtype=np.uint8), {}).observations


def test_ambiguous_match_is_rejected_by_score_margin(monkeypatch: Any) -> None:
    detector = OpenCvChampionDetector(
        TrackerConfig(ssim_threshold=0.3, ssim_margin=0.05), logging.getLogger("test")
    )
    query = np.zeros((20, 20, 3), dtype=np.uint8)
    monkeypatch.setattr(
        detector,
        "score_matches",
        lambda *_args: [("Aatrox", 0.90), ("Nami", 0.87)],
    )
    name, score = detector.find_best_match(query, {})
    assert name is None
    assert score == 0.90


def test_process_enforces_one_detection_per_champion(monkeypatch: Any) -> None:
    detector = OpenCvChampionDetector(TrackerConfig(), logging.getLogger("test"))
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    circles = np.array([[[15, 15, 5], [45, 45, 5]]], dtype=np.float32)
    scores = iter(
        [
            [("Aatrox", 0.95), ("Nami", 0.40)],
            [("Aatrox", 0.90), ("Nami", 0.40)],
        ]
    )
    monkeypatch.setattr(detector, "detect_red_circles", lambda _image: circles)
    monkeypatch.setattr(detector, "find_rectangle_center", lambda _image: None)
    monkeypatch.setattr(detector, "score_matches", lambda *_args: next(scores))

    detection = detector.process(image, {})
    assert [(item.champion_name, item.x, item.y) for item in detection.observations] == [
        ("Aatrox", 15, 15)
    ]
    assert detection.diagnostics.circles == 2
    assert detection.diagnostics.accepted == 1
    assert detection.diagnostics.duplicate == 1


def test_process_counts_ambiguous_and_below_threshold_candidates(monkeypatch: Any) -> None:
    detector = OpenCvChampionDetector(TrackerConfig(ssim_margin=0.05), logging.getLogger("test"))
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    circles = np.array([[[15, 15, 5], [45, 45, 5]]], dtype=np.float32)
    scores = iter(
        [
            [("Aatrox", 0.90), ("Nami", 0.88)],
            [("Aatrox", 0.20), ("Nami", 0.10)],
        ]
    )
    monkeypatch.setattr(detector, "detect_red_circles", lambda _image: circles)
    monkeypatch.setattr(detector, "score_matches", lambda *_args: next(scores))

    detection = detector.process(image, {})
    assert not detection.observations
    assert detection.diagnostics.ambiguous == 1
    assert detection.diagnostics.below_threshold == 1
