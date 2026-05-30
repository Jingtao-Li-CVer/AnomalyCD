"""Stage 1 entry point: bitemporal change detection."""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from img_io import read_img, write_img
from segment_anything_origin import SamAutomaticMaskGenerator, sam_model_registry

import config
from stage1_analysis import analyze_bitemporal_patch
from utils import (
    directory_has_files,
    iter_patch_offsets,
    list_event_dirs,
    resolve_event_paths,
    validate_event_paths,
)


def build_mask_generator(device: str) -> SamAutomaticMaskGenerator:
    """Load SAM and create an automatic mask generator for stage 1."""
    sam = sam_model_registry[config.SAM_MODEL_TYPE](
        checkpoint=config.SAM_CHECKPOINT
    ).to(device)
    return SamAutomaticMaskGenerator(sam, **config.STAGE1_SAM_PARAMS)


def process_event(
    event_name: str,
    mask_generator: SamAutomaticMaskGenerator,
    data_root: str,
    output_root: str,
    patch_size: int,
    skip_existing: bool,
) -> None:
    """Run stage-1 change detection for one event."""
    event_dir = os.path.join(data_root, event_name)
    anomaly_path, label_path, normal_path, _ = resolve_event_paths(event_dir)

    if not validate_event_paths(anomaly_path, normal_path, label_path):
        print(f"[Stage1] Skip invalid event: {event_name}")
        return

    save_dir = os.path.join(output_root, event_name)
    os.makedirs(save_dir, exist_ok=True)

    if skip_existing and directory_has_files(save_dir):
        print(f"[Stage1] Skip existing results: {save_dir}")
        return

    print(f"[Stage1] Processing {event_name}")

    normal_image = read_img(normal_path).astype(np.uint8)
    anomaly_image = read_img(anomaly_path).astype(np.uint8)

    height, width, _ = normal_image.shape
    change_map = np.zeros((height, width))

    for row, col in tqdm(
        list(iter_patch_offsets(height, width, patch_size)),
        desc=f"Stage1 patches ({event_name})",
    ):
        normal_patch = normal_image[row:row + patch_size, col:col + patch_size].copy()
        anomaly_patch = anomaly_image[row:row + patch_size, col:col + patch_size].copy()

        patch_change_map = analyze_bitemporal_patch(
            mask_generator, normal_patch, anomaly_patch
        )
        change_map[row:row + patch_size, col:col + patch_size] = patch_change_map

    write_img(change_map, os.path.join(save_dir, "change_map_continuous.tif"))

    threshold = np.quantile(change_map, config.STAGE1_CHANGE_QUANTILE)
    binary_change_map = (change_map >= threshold).astype(np.float32)
    cv2.imwrite(
        os.path.join(save_dir, "change_map_binary.png"),
        binary_change_map * 255,
    )
    print(f"[Stage1] Finished {event_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnomalyCD Stage 1: change detection")
    parser.add_argument("--data-root", default=config.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default=config.STAGE1_OUTPUT_DIR)
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)
    parser.add_argument("--patch-size", type=int, default=config.STAGE1_PATCH_SIZE)
    parser.add_argument("--event", default=None, help="Process a single event directory")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip events whose output directory already contains files",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Re-run events even when outputs already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("GDAL_CACHEMAX", "40000")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AnomalyCD inference.")

    mask_generator = build_mask_generator(args.device)
    event_names = [args.event] if args.event else list_event_dirs(args.data_root)

    for event_name in tqdm(event_names, desc="Stage1 events"):
        process_event(
            event_name=event_name,
            mask_generator=mask_generator,
            data_root=args.data_root,
            output_root=args.output_root,
            patch_size=args.patch_size,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
