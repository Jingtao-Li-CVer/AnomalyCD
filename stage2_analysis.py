"""Stage 2: anomaly change detection analysis."""

from __future__ import annotations

import numpy as np
import torch

from utils import max_cosine_dissimilarity


def extract_patch_features(mask_generator, image_patch: np.ndarray) -> torch.Tensor:
    """Extract SAM image-encoder features for one image patch."""
    mask_generator.predictor.set_image(image_patch)
    return mask_generator.predictor.features


def analyze_anomaly_change_patch(
    mask_generator,
    temporal_patches: list[np.ndarray],
    change_map_patch: np.ndarray,
    min_mask_pixels: int = 1500,
) -> np.ndarray:
    """
    Distinguish abnormal changes from normal temporal changes.

    For each connected region in the stage-1 change map, compare SAM features
    across all temporal images and assign a dissimilarity score relative to
    the anomaly image (last temporal term).
    """
    anomaly_change_map = np.zeros(change_map_patch.shape)
    patch_features = [
        extract_patch_features(mask_generator, patch) for patch in temporal_patches
    ]

    unique_scores = np.unique(change_map_patch)
    feature_grid_size = 64

    for score in unique_scores:
        if score == 0:
            continue

        region_mask = (change_map_patch == score).astype(np.uint8)
        region_area_ratio = region_mask.sum() / region_mask.size
        if region_area_ratio > 0.9:
            continue
        if region_mask.sum() < min_mask_pixels:
            continue

        row_indices, col_indices = np.where(region_mask)
        feature_rows = (
            row_indices / change_map_patch.shape[0] * feature_grid_size
        ).astype(np.uint8).tolist()
        feature_cols = (
            col_indices / change_map_patch.shape[1] * feature_grid_size
        ).astype(np.uint8).tolist()

        temporal_region_features = [
            features[0, :, feature_rows, feature_cols].mean(1)
            for features in patch_features
        ]
        dissimilarity = max_cosine_dissimilarity(temporal_region_features)
        anomaly_change_map[row_indices, col_indices] = dissimilarity

    return anomaly_change_map
