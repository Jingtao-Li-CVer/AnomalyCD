"""Evaluate AnomalyCD predictions with TPR, weighted precision, and F1."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from img_io import read_img
import config
from eval_metrics import (
    calculate_category_averages,
    calculate_f1,
    calculate_f1_by_category,
    calculate_tpr,
    calculate_weighted_precision,
    extract_event_category_id,
    normalize_map,
    threshold_prediction_map,
)
from utils import binarize_label, list_event_dirs, resolve_event_paths


def find_prediction_files(result_dir: str) -> Tuple[str, str]:
    """Locate stage-2 change-map and AnomalyCD result files."""
    anomaly_files = sorted(glob.glob(os.path.join(result_dir, "AnomalyCD_map*.tif")))
    change_files = sorted(glob.glob(os.path.join(result_dir, "change_map_filtered*.tif")))

    if not anomaly_files:
        raise FileNotFoundError(f"Missing AnomalyCD_map*.tif in {result_dir}")
    if not change_files:
        raise FileNotFoundError(f"Missing change_map_filtered*.tif in {result_dir}")

    return anomaly_files[0], change_files[0]


def init_category_records(num_categories: int) -> Dict[str, List[List[float]]]:
    """Create empty metric containers for each event category."""
    empty = [[] for _ in range(num_categories)]
    return {
        "tpr_change_map": [list(bucket) for bucket in empty],
        "tpr_anomalycd": [list(bucket) for bucket in empty],
        "precision_change_map": [list(bucket) for bucket in empty],
        "precision_anomalycd": [list(bucket) for bucket in empty],
    }


def evaluate_event(
    event_name: str,
    data_root: str,
    result_root: str,
    quantile_threshold: float,
    area_ratio: float,
    background_weight: float,
    normalize_anomaly_map: bool,
) -> Optional[dict]:
    """Evaluate one event and return per-event metric records."""
    category_id = extract_event_category_id(event_name)
    if category_id is None or category_id <= 0:
        return None

    event_dir = os.path.join(data_root, event_name)
    _, label_path, _, _ = resolve_event_paths(event_dir)
    result_dir = os.path.join(result_root, event_name)

    if not os.path.isdir(result_dir):
        return None

    try:
        anomaly_map_path, change_map_path = find_prediction_files(result_dir)
    except FileNotFoundError:
        return None

    ground_truth = binarize_label(read_img(label_path))
    change_map = read_img(change_map_path).copy()
    anomaly_map = read_img(anomaly_map_path).copy()

    change_binary = threshold_prediction_map(
        change_map,
        quantile_threshold=quantile_threshold,
        area_ratio=area_ratio,
        apply_morphology=False,
    )

    if normalize_anomaly_map:
        anomaly_map = normalize_map(anomaly_map)

    anomaly_binary = threshold_prediction_map(
        anomaly_map,
        quantile_threshold=quantile_threshold,
        area_ratio=area_ratio,
        apply_morphology=True,
    )

    change_tpr = calculate_tpr(ground_truth, change_binary)
    anomaly_tpr = calculate_tpr(ground_truth, anomaly_binary)
    change_precision = calculate_weighted_precision(
        ground_truth, change_binary, background_weight
    )
    anomaly_precision = calculate_weighted_precision(
        ground_truth, anomaly_binary, background_weight
    )

    return {
        "event_name": event_name,
        "category_id": category_id,
        "normal_terms": len(os.listdir(event_dir)) - 2,
        "change_tpr": change_tpr,
        "change_precision": change_precision,
        "change_f1": calculate_f1(change_tpr, change_precision),
        "anomaly_tpr": anomaly_tpr,
        "anomaly_precision": anomaly_precision,
        "anomaly_f1": calculate_f1(anomaly_tpr, anomaly_precision),
    }


def save_records_csv(records: List[dict], output_csv: str) -> None:
    """Save per-event evaluation records to CSV."""
    fieldnames = [
        "event_name",
        "category_id",
        "normal_terms",
        "change_tpr",
        "change_precision",
        "change_f1",
        "anomaly_tpr",
        "anomaly_precision",
        "anomaly_f1",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_category_summary(
    quantile_threshold: float,
    records: List[dict],
    num_categories: int,
) -> None:
    """Aggregate and print category-wise evaluation results."""
    category_records = init_category_records(num_categories)

    for record in records:
        bucket_index = record["category_id"] - 1
        category_records["tpr_change_map"][bucket_index].append(record["change_tpr"])
        category_records["tpr_anomalycd"][bucket_index].append(record["anomaly_tpr"])
        category_records["precision_change_map"][bucket_index].append(
            record["change_precision"]
        )
        category_records["precision_anomalycd"][bucket_index].append(
            record["anomaly_precision"]
        )

    change_tpr_avg = calculate_category_averages(category_records["tpr_change_map"])
    anomaly_tpr_avg = calculate_category_averages(category_records["tpr_anomalycd"])
    change_precision_avg = calculate_category_averages(
        category_records["precision_change_map"]
    )
    anomaly_precision_avg = calculate_category_averages(
        category_records["precision_anomalycd"]
    )
    change_f1_avg = calculate_f1_by_category(change_tpr_avg, change_precision_avg)
    anomaly_f1_avg = calculate_f1_by_category(anomaly_tpr_avg, anomaly_precision_avg)

    print(f"Quantile threshold: {quantile_threshold}")
    print(f"Evaluated events: {len(records)}")
    print("TPR (change map):", change_tpr_avg)
    print("TPR (AnomalyCD):", anomaly_tpr_avg)
    print("Precision (change map):", change_precision_avg)
    print("Precision (AnomalyCD):", anomaly_precision_avg)
    print("F1 (change map):", change_f1_avg)
    print("F1 (AnomalyCD):", anomaly_f1_avg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate AnomalyCD stage-2 outputs with TPR, precision, and F1"
    )
    parser.add_argument("--data-root", default=config.DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", default=config.STAGE2_OUTPUT_DIR)
    parser.add_argument("--max-events", type=int, default=config.EVAL_MAX_EVENTS)
    parser.add_argument(
        "--quantile-threshold",
        type=float,
        default=config.EVAL_QUANTILE_THRESH,
        help="Quantile threshold used to binarize prediction maps",
    )
    parser.add_argument(
        "--area-ratio",
        type=float,
        default=config.EVAL_AREA_RATIO,
        help="Minimum connected-component area ratio for AnomalyCD post-processing",
    )
    parser.add_argument(
        "--background-weight",
        type=float,
        default=config.EVAL_BACKGROUND_WEIGHT,
        help="Background false-positive weight used in precision",
    )
    parser.add_argument(
        "--num-categories",
        type=int,
        default=config.EVAL_NUM_CATEGORIES,
        help="Number of event categories aggregated in the summary",
    )
    parser.add_argument(
        "--normalize-anomaly-map",
        action="store_true",
        help="Min-max normalize AnomalyCD maps before thresholding",
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
            area_ratio=args.area_ratio,
            background_weight=args.background_weight,
            normalize_anomaly_map=args.normalize_anomaly_map,
        )
        if record is None:
            continue

        all_records.append(record)
        print(
            f"{record['event_name']}: "
            f"change_tpr={record['change_tpr']:.4f}, "
            f"anomaly_tpr={record['anomaly_tpr']:.4f}"
        )

    if not all_records:
        raise RuntimeError("No valid evaluation records were produced.")

    print_category_summary(args.quantile_threshold, all_records, args.num_categories)

    if args.output_csv:
        save_records_csv(all_records, args.output_csv)
        print(f"Saved per-event records to {args.output_csv}")


if __name__ == "__main__":
    main()
