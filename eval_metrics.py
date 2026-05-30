"""Evaluation metrics for AnomalyCD."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np
import skimage
from skimage import morphology


def extract_event_category_id(event_name: str) -> Optional[int]:
    """Extract the leading category id from an event directory name."""
    match = re.match(r"^(\d+)", event_name)
    if not match:
        return None
    return int(match.group(1))


def remove_small_regions(binary_map: np.ndarray, area_threshold: int) -> np.ndarray:
    """Remove connected components smaller than the area threshold."""
    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_map, connectivity=8
    )
    cleaned = np.zeros(binary_map.shape, dtype=np.uint8)
    for component_id in range(num_components - 1):
        if stats[component_id + 1, cv2.CC_STAT_AREA] >= area_threshold:
            cleaned[labels == (component_id + 1)] = 1
    return cleaned


def threshold_prediction_map(
    prediction_map: np.ndarray,
    quantile_threshold: float = 0.9,
    area_ratio: float = 0.0003,
    apply_morphology: bool = False,
) -> np.ndarray:
    """
    Binarize a continuous prediction map using a quantile threshold.

    When apply_morphology is True, perform closing and small-region removal
    after thresholding, following the original evaluation protocol.
    """
    binary_map = prediction_map.copy()
    threshold = np.quantile(binary_map, quantile_threshold)

    if threshold > 0:
        binary_map[binary_map >= threshold] = 1
        binary_map[binary_map < threshold] = 0
    else:
        binary_map[binary_map > 0] = 1

    if not apply_morphology:
        return binary_map

    min_region_area = int(binary_map.shape[0] * binary_map.shape[1] * area_ratio)
    binary_map = binary_map.astype(bool)
    original_shape = binary_map.shape

    binary_map = skimage.transform.resize(binary_map, (512, 512))
    binary_map = morphology.binary_closing(binary_map, morphology.disk(3))
    binary_map = skimage.transform.resize(binary_map, original_shape)
    return remove_small_regions(binary_map.astype(np.uint8), min_region_area)


def threshold_prediction_map_by_value(
    prediction_map: np.ndarray,
    fixed_threshold: float = 0.08,
    area_ratio: float = 0.0003,
    apply_morphology: bool = False,
) -> np.ndarray:
    """
    Binarize a continuous prediction map using a fixed threshold.

    When apply_morphology is True, perform closing and small-region removal
    after thresholding, following the original evaluation protocol.
    """
    binary_map = prediction_map.copy()
    binary_map[binary_map >= fixed_threshold] = 1
    binary_map[binary_map < fixed_threshold] = 0

    if not apply_morphology:
        return binary_map

    min_region_area = int(binary_map.shape[0] * binary_map.shape[1] * area_ratio)
    binary_map = binary_map.astype(bool)
    original_shape = binary_map.shape

    binary_map = skimage.transform.resize(binary_map, (512, 512))
    binary_map = morphology.binary_closing(binary_map, morphology.disk(3))
    binary_map = skimage.transform.resize(binary_map, original_shape)
    return remove_small_regions(binary_map.astype(np.uint8), min_region_area)


def normalize_map(prediction_map: np.ndarray) -> np.ndarray:
    """Min-max normalize a prediction map to [0, 1]."""
    value_range = prediction_map.max() - prediction_map.min()
    if value_range == 0:
        return np.zeros_like(prediction_map, dtype=np.float32)
    return (prediction_map - prediction_map.min()) / value_range


def calculate_tpr(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """Compute true positive rate (recall)."""
    true_positives = np.sum(np.logical_and(ground_truth == 1, prediction == 1))
    actual_positives = np.sum(ground_truth == 1)
    if actual_positives == 0:
        return 0.0
    return float(true_positives / actual_positives)


def calculate_weighted_precision(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    background_weight: float = 0.1,
) -> float:
    """Compute precision with down-weighted false positives on background."""
    true_positives = np.sum(np.logical_and(ground_truth == 1, prediction == 1))
    background_false_positives = np.sum(np.logical_and(ground_truth == 0, prediction == 1))
    weighted_denominator = true_positives + background_weight * background_false_positives
    if weighted_denominator == 0:
        return 0.0
    return float(true_positives / weighted_denominator)


def calculate_f1(true_positive_rate: float, precision: float) -> float:
    """Compute F1 score from TPR and precision."""
    if true_positive_rate + precision == 0:
        return 0.0
    return float(2 * true_positive_rate * precision / (true_positive_rate + precision))


def calculate_category_averages(records_by_category: Sequence[Sequence[float]]) -> List[float]:
    """Compute per-category averages from nested metric records."""
    averages = []
    for category_records in records_by_category:
        if category_records:
            averages.append(float(sum(category_records) / len(category_records)))
        else:
            averages.append(0.0)
    return averages


def calculate_f1_by_category(
    tpr_averages: Iterable[float],
    precision_averages: Iterable[float],
) -> List[float]:
    """Compute F1 for each category from averaged TPR and precision."""
    return [
        calculate_f1(tpr, precision)
        for tpr, precision in zip(tpr_averages, precision_averages)
    ]
