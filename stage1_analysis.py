"""Stage 1: bitemporal change detection analysis."""

import numpy as np
import torch.nn.functional as F

from utils import compute_change_map_from_masks


def analyze_bitemporal_patch(
    mask_generator,
    normal_patch: np.ndarray,
    anomaly_patch: np.ndarray,
    filter_small_area: bool = True,
) -> np.ndarray:
    """
    Detect pixel-level changes between a normal patch and an anomaly patch.

    SAM masks are generated on both images; change scores are computed from
    SAM image-encoder features and merged by taking the element-wise maximum.
    """
    normal_masks = mask_generator.generate(normal_patch)

    mask_generator.predictor.set_image(normal_patch)
    normal_features = mask_generator.predictor.features
    normal_features = F.interpolate(
        normal_features, (normal_patch.shape[0], normal_patch.shape[1])
    )

    anomaly_masks = mask_generator.generate(anomaly_patch)

    mask_generator.predictor.set_image(anomaly_patch)
    anomaly_features = mask_generator.predictor.features
    anomaly_features = F.interpolate(
        anomaly_features, (normal_patch.shape[0], normal_patch.shape[1])
    )

    change_from_anomaly_masks = compute_change_map_from_masks(
        anomaly_masks, normal_features, anomaly_features, filter_small_area
    )
    change_from_normal_masks = compute_change_map_from_masks(
        normal_masks, normal_features, anomaly_features, filter_small_area
    )

    return np.maximum(change_from_anomaly_masks, change_from_normal_masks)
