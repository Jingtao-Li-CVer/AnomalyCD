"""Shared utilities for AnomalyCD."""

from __future__ import annotations

import os
import re
from typing import Optional

import cv2
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve


def list_event_dirs(data_root: str) -> list[str]:
    """List sorted event directories under the dataset root."""
    event_names = os.listdir(data_root)
    event_names.sort()
    return event_names


def resolve_event_paths(event_dir: str) -> tuple[str, str, str, list[str]]:
    """
    Resolve image paths for one event directory.

    Expected layout (sorted filenames):
      term_names[0]  -> anomaly image
      term_names[1]  -> label
      term_names[2:] -> normal temporal images
    """
    term_names = os.listdir(event_dir)
    term_names.sort()

    anomaly_path = os.path.join(event_dir, term_names[0])
    label_path = os.path.join(event_dir, term_names[1])
    normal_paths = [
        os.path.join(event_dir, term_name)
        for term_name in term_names[2:]
    ]
    latest_normal_path = os.path.join(event_dir, term_names[-1])

    return anomaly_path, label_path, latest_normal_path, normal_paths


def validate_event_paths(anomaly_path: str, normal_path: str, label_path: str) -> bool:
    """Return True when filenames match the expected naming convention."""
    return (
        "anomaly" in os.path.basename(anomaly_path)
        and "normal" in os.path.basename(normal_path)
        and "label" in os.path.basename(label_path)
    )


def binarize_label(label: np.ndarray) -> np.ndarray:
    """
    Convert multi-class annotation rasters to a binary anomaly mask.

    Raw labels use different positive integers for distinct anomaly categories.
    For AUC / TPR / precision evaluation, value 1 is treated as background and
    all other positive values are merged into the anomaly class.
    """
    binary_label = label.copy()
    binary_label[binary_label == 1] = 0
    binary_label[binary_label > 0] = 1
    return binary_label


def directory_has_files(directory: str) -> bool:
    """Return True if the directory contains at least one file."""
    for entry in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, entry)):
            return True
    return False


def list_tif_files(directory: str) -> list[str]:
    """List .tif files in a directory."""
    return sorted(
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".tif")
    )


def get_patch_size(height: int, width: int, default_size: int = 2048) -> int:
    """Select patch size based on image dimensions."""
    if height < 2048 or width < 2048:
        return 1024
    return default_size


def iter_patch_offsets(height: int, width: int, patch_size: int):
    """Yield top-left (y, x) coordinates for sliding-window inference."""
    for x in range(0, width, patch_size):
        if x + patch_size > width:
            x = width - patch_size
        for y in range(0, height, patch_size):
            if y + patch_size > height:
                y = height - patch_size
            yield y, x


def calculate_auc(prediction_map: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute ROC-AUC between a continuous prediction map and binary labels."""
    prediction_flat = prediction_map.flatten()
    ground_truth_flat = ground_truth.flatten()
    false_positive_rate, true_positive_rate, _ = roc_curve(
        ground_truth_flat, prediction_flat
    )
    return auc(false_positive_rate, true_positive_rate)


def cosine_similarity(vector_a: torch.Tensor, vector_b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two feature vectors."""
    norm_a = torch.norm(vector_a, dim=0)
    norm_b = torch.norm(vector_b, dim=0)
    dot_product = torch.dot(vector_a, vector_b)
    return dot_product / (norm_a * norm_b + 1e-8)


def compute_change_map_from_masks(
    sam_masks: list,
    reference_features: torch.Tensor,
    target_features: torch.Tensor,
    filter_small_area: bool = True,
    min_mask_area: int = 3000,
) -> np.ndarray:
    """
    Compute a continuous change map from SAM instance masks.

    For each SAM mask, compare mean SAM image-encoder features between two
    temporal images and assign a change score of 1 - cosine_similarity.
    """
    if not sam_masks:
        return None

    sorted_masks = sorted(sam_masks, key=lambda mask: mask["area"], reverse=True)
    change_map = np.zeros(sorted_masks[0]["segmentation"].shape)

    image_area = (
        sorted_masks[0]["segmentation"].shape[0]
        * sorted_masks[0]["segmentation"].shape[1]
    )

    for mask_info in sorted_masks:
        mask = mask_info["segmentation"]
        if mask_info["area"] / image_area > 0.9:
            continue
        if filter_small_area and mask_info["area"] < min_mask_area:
            continue

        row_indices, col_indices = np.where(mask)
        reference_feature = reference_features[0, :, row_indices, col_indices].mean(1)
        target_feature = target_features[0, :, row_indices, col_indices].mean(1)

        change_score = 1 - cosine_similarity(reference_feature, target_feature).item()
        change_score = max(0.0, min(change_score, 1.0))
        change_map[row_indices, col_indices] = change_score

    return change_map


def max_cosine_dissimilarity(feature_vectors: list[torch.Tensor]) -> float:
    """
    Compare the last temporal feature vector against all previous ones.

    Returns 1 - max(cosine_similarity), i.e. the largest dissimilarity score.
    """
    feature_tensor = torch.stack(feature_vectors)
    anomaly_feature = feature_tensor[-1]
    dot_products = torch.mv(feature_tensor[:-1], anomaly_feature)
    norms = torch.norm(feature_tensor[:-1], dim=1) * torch.norm(anomaly_feature)
    cosine_similarities = dot_products / norms
    return 1 - cosine_similarities.max().item()


def extract_auc_from_filename(filename: str) -> Optional[float]:
    """Extract a floating-point AUC value embedded in a result filename."""
    match = re.search(r"(\d+\.\d+)", filename)
    if match:
        return float(match.group(1))
    return None
