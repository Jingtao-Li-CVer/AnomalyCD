"""Quick smoke test for AnomalyCD on one image patch."""

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


def main() -> None:
    device = "cuda:1"
    event_dir = "/media/data2/ljt/全球地表异常数据集/data/1_赤道几内亚爆炸-20210313_30"
    term_names = sorted(os.listdir(event_dir))
    anomaly_path = os.path.join(event_dir, term_names[0])
    normal_path = os.path.join(event_dir, term_names[-1])
    normal_paths = [os.path.join(event_dir, name) for name in term_names[2:]]

    print("Loading SAM...")
    sam = sam_model_registry[config.SAM_MODEL_TYPE](checkpoint=config.SAM_CHECKPOINT).to(device)
    stage1_generator = SamAutomaticMaskGenerator(sam, **config.STAGE1_SAM_PARAMS)

    patch_size = 1024
    normal_image = read_img(normal_path).astype(np.uint8)
    anomaly_image = read_img(anomaly_path).astype(np.uint8)
    normal_patch = normal_image[:patch_size, :patch_size].copy()
    anomaly_patch = anomaly_image[:patch_size, :patch_size].copy()

    print("Stage1 patch inference...")
    change_patch = analyze_bitemporal_patch(stage1_generator, normal_patch, anomaly_patch)
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
    temporal_images = [read_img(path).astype(np.uint8) for path in normal_paths] + [
        anomaly_image
    ]
    temporal_patches = [image[:patch_size, :patch_size].copy() for image in temporal_images]
    filtered_change_patch = change_patch.copy()
    filtered_change_patch[
        filtered_change_patch < np.quantile(filtered_change_patch, 0.7)
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
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
