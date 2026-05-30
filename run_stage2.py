"""Stage 2 entry point: anomaly change detection."""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from img_io import read_img, write_img
from segment_anything_origin import SamAutomaticMaskGenerator, sam_model_registry

import config
from stage2_analysis import analyze_anomaly_change_patch
from utils import (
    binarize_label,
    calculate_auc,
    get_patch_size,
    iter_patch_offsets,
    list_event_dirs,
    list_tif_files,
    resolve_event_paths,
)


def build_mask_generator(device: str) -> SamAutomaticMaskGenerator:
    """Load SAM and create an automatic mask generator for stage 2."""
    sam = sam_model_registry[config.SAM_MODEL_TYPE](
        checkpoint=config.SAM_CHECKPOINT
    ).to(device)
    return SamAutomaticMaskGenerator(sam, **config.STAGE2_SAM_PARAMS)


def load_stage1_change_map(stage1_dir: str) -> np.ndarray:
    """Load the stage-1 change map and apply the stage-2 filtering quantile."""
    tif_files = list_tif_files(stage1_dir)
    if not tif_files:
        raise FileNotFoundError(f"No stage-1 .tif outputs found in {stage1_dir}")

    change_map = read_img(tif_files[0])
    if len(tif_files) == 2:
        change_map = np.maximum(change_map, read_img(tif_files[1]))

    threshold = np.quantile(change_map, config.STAGE2_CHANGE_MAP_QUANTILE)
    filtered_change_map = change_map.copy()
    filtered_change_map[filtered_change_map < threshold] = 0
    return filtered_change_map


def process_event(
    event_name: str,
    mask_generator: SamAutomaticMaskGenerator,
    data_root: str,
    stage1_root: str,
    output_root: str,
    skip_existing: bool,
) -> None:
    """Run stage-2 anomaly change detection for one event."""
    event_dir = os.path.join(data_root, event_name)
    anomaly_path, label_path, _, normal_paths = resolve_event_paths(event_dir)

    save_dir = os.path.join(output_root, event_name)
    os.makedirs(save_dir, exist_ok=True)

    if skip_existing and os.path.exists(
        os.path.join(save_dir, "AnomalyCD_map.tif")
    ):
        print(f"[Stage2] Skip existing results: {save_dir}")
        return

    print(f"[Stage2] Processing {event_name}")

    temporal_paths = normal_paths + [anomaly_path]
    temporal_images = [read_img(path) for path in temporal_paths]
    height, width, _ = temporal_images[-1].shape
    patch_size = get_patch_size(height, width)

    stage1_dir = os.path.join(stage1_root, event_name)
    change_map = load_stage1_change_map(stage1_dir)
    anomaly_change_map = np.zeros((height, width))

    for row, col in tqdm(
        list(iter_patch_offsets(height, width, patch_size)),
        desc=f"Stage2 patches ({event_name})",
    ):
        temporal_patches = [
            image[row:row + patch_size, col:col + patch_size, :].copy()
            for image in temporal_images
        ]
        change_map_patch = change_map[row:row + patch_size, col:col + patch_size].copy()

        patch_anomaly_map = analyze_anomaly_change_patch(
            mask_generator,
            temporal_patches,
            change_map_patch,
            min_mask_pixels=config.STAGE2_MIN_MASK_PIXELS,
        )
        anomaly_change_map[row:row + patch_size, col:col + patch_size] = patch_anomaly_map

    if event_name.startswith("0_"):
        write_img(change_map, os.path.join(save_dir, "change_map_filtered.tif"))
        write_img(anomaly_change_map, os.path.join(save_dir, "AnomalyCD_map.tif"))
        print(f"[Stage2] Saved normal-event outputs for {event_name}")
        return

    ground_truth = binarize_label(read_img(label_path))

    change_map_auc = calculate_auc(change_map, ground_truth)
    anomaly_map_auc = calculate_auc(anomaly_change_map, ground_truth)
    print(f"[Stage2] Change-map AUC: {change_map_auc:.6f}")
    print(f"[Stage2] AnomalyCD AUC: {anomaly_map_auc:.6f}")

    write_img(
        change_map,
        os.path.join(save_dir, f"change_map_filtered_{change_map_auc:.6f}.tif"),
    )
    write_img(
        anomaly_change_map,
        os.path.join(save_dir, f"AnomalyCD_map_{anomaly_map_auc:.6f}.tif"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnomalyCD Stage 2: anomaly change")
    parser.add_argument("--data-root", default=config.DEFAULT_DATA_ROOT)
    parser.add_argument("--stage1-root", default=config.STAGE1_OUTPUT_DIR)
    parser.add_argument("--output-root", default=config.STAGE2_OUTPUT_DIR)
    parser.add_argument("--device", default=config.DEFAULT_DEVICE)
    parser.add_argument("--event", default=None, help="Process a single event directory")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip events when AnomalyCD_map.tif already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for AnomalyCD inference.")

    mask_generator = build_mask_generator(args.device)
    event_names = [args.event] if args.event else list_event_dirs(args.data_root)

    for event_name in tqdm(event_names, desc="Stage2 events"):
        stage1_dir = os.path.join(args.stage1_root, event_name)
        if not os.path.isdir(stage1_dir):
            print(f"[Stage2] Missing stage-1 output, skip: {event_name}")
            continue

        process_event(
            event_name=event_name,
            mask_generator=mask_generator,
            data_root=args.data_root,
            stage1_root=args.stage1_root,
            output_root=args.output_root,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
