"""Evaluate AnomalyCD with per-event recall, precision, and F1."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import List, Optional, Tuple

from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from img_io import read_img
import config
from eval_metrics import (
    calculate_f1,
    calculate_tpr,
    calculate_weighted_precision,
    normalize_map,
    threshold_prediction_map,
    threshold_prediction_map_by_value,
)
from utils import (
    binarize_label,
    is_evaluable_anomaly_event,
    list_event_dirs,
    resolve_event_paths,
)


def find_prediction_files(result_dir: str) -> Tuple[str, str]:
    """Locate stage-1 change map and stage-2 AnomalyCD result files."""
    anomaly_files = sorted(glob.glob(os.path.join(result_dir, "AnomalyCD_map*.tif")))
    change_files = sorted(glob.glob(os.path.join(result_dir, "change_map_filtered*.tif")))

    if not anomaly_files:
        raise FileNotFoundError(f"Missing AnomalyCD_map*.tif in {result_dir}")
    if not change_files:
        raise FileNotFoundError(f"Missing change_map_filtered*.tif in {result_dir}")

    return change_files[0], anomaly_files[0]


def evaluate_event(
    event_name: str,
    data_root: str,
    result_root: str,
    quantile_threshold: float,
    stage2_fixed_threshold: float,
    area_ratio: float,
    background_weight: float,
    normalize_anomaly_map: bool,
) -> Optional[dict]:
    """Evaluate one anomaly event and return stage-1 / stage-2 metrics."""
    if not is_evaluable_anomaly_event(event_name):
        return None

    event_dir = os.path.join(data_root, event_name)
    _, label_path, _, _ = resolve_event_paths(event_dir, event_name)
    if label_path is None:
        return None
    result_dir = os.path.join(result_root, event_name)

    if not os.path.isdir(result_dir):
        return None

    try:
        stage1_map_path, stage2_map_path = find_prediction_files(result_dir)
    except FileNotFoundError:
        return None

    ground_truth = binarize_label(read_img(label_path))
    stage1_map = read_img(stage1_map_path).copy()
    stage2_map = read_img(stage2_map_path).copy()

    stage1_binary = threshold_prediction_map(
        stage1_map,
        quantile_threshold=quantile_threshold,
        area_ratio=area_ratio,
        apply_morphology=False,
    )

    if normalize_anomaly_map:
        stage2_map = normalize_map(stage2_map)

    stage2_binary = threshold_prediction_map_by_value(
        stage2_map,
        fixed_threshold=stage2_fixed_threshold,
        area_ratio=area_ratio,
        apply_morphology=True,
    )

    stage1_recall = calculate_tpr(ground_truth, stage1_binary)
    stage1_precision = calculate_weighted_precision(
        ground_truth, stage1_binary, background_weight
    )
    stage1_f1 = calculate_f1(stage1_recall, stage1_precision)

    stage2_recall = calculate_tpr(ground_truth, stage2_binary)
    stage2_precision = calculate_weighted_precision(
        ground_truth, stage2_binary, background_weight
    )
    stage2_f1 = calculate_f1(stage2_recall, stage2_precision)

    return {
        "event_name": event_name,
        "stage1_recall": stage1_recall,
        "stage1_precision": stage1_precision,
        "stage1_f1": stage1_f1,
        "stage2_recall": stage2_recall,
        "stage2_precision": stage2_precision,
        "stage2_f1": stage2_f1,
    }


def print_event_metrics(record: dict) -> None:
    """Print per-event evaluation results."""
    print(record["event_name"])
    print(
        "  Stage1  recall={:.4f}  precision={:.4f}  F1={:.4f}".format(
            record["stage1_recall"],
            record["stage1_precision"],
            record["stage1_f1"],
        )
    )
    print(
        "  Stage2  recall={:.4f}  precision={:.4f}  F1={:.4f}".format(
            record["stage2_recall"],
            record["stage2_precision"],
            record["stage2_f1"],
        )
    )


def save_records_csv(records: List[dict], output_csv: str) -> None:
    """Save per-event evaluation records to CSV."""
    fieldnames = [
        "event_name",
        "stage1_recall",
        "stage1_precision",
        "stage1_f1",
        "stage2_recall",
        "stage2_precision",
        "stage2_f1",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AnomalyCD with per-event recall, precision, and F1"
    )
    parser.add_argument("--data-root", default=config.DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", default=config.STAGE2_OUTPUT_DIR)
    parser.add_argument("--max-events", type=int, default=config.EVAL_MAX_EVENTS)
    parser.add_argument(
        "--quantile-threshold",
        type=float,
        default=config.EVAL_QUANTILE_THRESH,
        help="Quantile threshold for Stage 1 binarization",
    )
    parser.add_argument(
        "--stage2-threshold",
        type=float,
        default=config.EVAL_STAGE2_FIXED_THRESH,
        help="Fixed threshold for Stage 2 AnomalyCD map binarization",
    )
    parser.add_argument(
        "--area-ratio",
        type=float,
        default=config.EVAL_AREA_RATIO,
    )
    parser.add_argument(
        "--background-weight",
        type=float,
        default=config.EVAL_BACKGROUND_WEIGHT,
    )
    parser.add_argument(
        "--normalize-anomaly-map",
        action="store_true",
        help="Min-max normalize stage-2 maps before thresholding",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save per-event evaluation records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_names = list_event_dirs(args.data_root)[: args.max_events]
    all_records: List[dict] = []

    for event_name in tqdm(event_names, desc="Evaluating events"):
        record = evaluate_event(
            event_name=event_name,
            data_root=args.data_root,
            result_root=args.result_root,
            quantile_threshold=args.quantile_threshold,
            stage2_fixed_threshold=args.stage2_threshold,
            area_ratio=args.area_ratio,
            background_weight=args.background_weight,
            normalize_anomaly_map=args.normalize_anomaly_map,
        )
        if record is None:
            continue

        all_records.append(record)
        print_event_metrics(record)

    if not all_records:
        raise RuntimeError("No valid evaluation records were produced.")

    if args.output_csv:
        save_records_csv(all_records, args.output_csv)
        print(f"Saved records to {args.output_csv}")


if __name__ == "__main__":
    main()
