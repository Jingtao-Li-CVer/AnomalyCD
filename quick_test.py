"""Quick test for AnomalyCD on one image patch."""

import os
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from img_io import read_img
from segment_anything_origin import SamAutomaticMaskGenerator, sam_model_registry

import config
from stage1_analysis import analyze_bitemporal_patch
from stage2_analysis import analyze_anomaly_change_patch
from utils import resolve_event_paths


def main() -> None:
    device = config.DEFAULT_DEVICE
    event_name = "1_赤道几内亚爆炸-20210313_30"
    event_dir = os.path.join(config.DEFAULT_DATA_ROOT, event_name)
    monitor_path, _, reference_path, historical_paths = resolve_event_paths(
        event_dir, event_name
    )

    print("Loading SAM...")
    sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT).to(device)
    stage1_generator = SamAutomaticMaskGenerator(sam, **config.STAGE1_SAM_PARAMS)

    patch_size = 1024
    reference_image = read_img(reference_path).astype(np.uint8)
    monitor_image = read_img(monitor_path).astype(np.uint8)
    reference_patch = reference_image[:patch_size, :patch_size].copy()
    monitor_patch = monitor_image[:patch_size, :patch_size].copy()

    print("Stage1 patch inference...")
    change_patch = analyze_bitemporal_patch(stage1_generator, reference_patch, monitor_patch)
    print(
        "Stage1 output:",
        change_patch.shape,
        float(change_patch.min()),
        float(change_patch.max()),
    )

    stage2_generator = SamAutomaticMaskGenerator(
        sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT).to(
            device
        ),
        **config.STAGE2_SAM_PARAMS,
    )
    temporal_images = [read_img(path).astype(np.uint8) for path in historical_paths] + [
        monitor_image
    ]
    temporal_patches = [image[:patch_size, :patch_size].copy() for image in temporal_images]
    filtered_change_patch = change_patch.copy()
    filtered_change_patch[
        filtered_change_patch < np.quantile(filtered_change_patch, config.STAGE2_CHANGE_MAP_QUANTILE)
    ] = 0

    print("Stage2 patch inference...")
    anomaly_patch_map = analyze_anomaly_change_patch(
        stage2_generator, temporal_patches, filtered_change_patch
    )
    print(
        "Stage2 output:",
        anomaly_patch_map.shape,
        float(anomaly_patch_map.min()),
        float(anomaly_patch_map.max()),
    )
    print("QUICK TEST PASSED")


if __name__ == "__main__":
    main()
