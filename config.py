"""AnomalyCD configuration."""

import os

# Package directory (all runtime dependencies live under this folder)
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model weights
SAM_CHECKPOINT = "/media/data1/ljt/weights/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

# Dataset paths
DEFAULT_DATA_ROOT = "/media/data2/ljt/全球地表异常数据集/data"
DEFAULT_OUTPUT_ROOT = "/media/data2/ljt/全球地表异常数据集/AnomalyCD"

STAGE1_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "stage1_change_detection")
STAGE2_OUTPUT_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "stage2_anomaly_change")

# Stage 1: bitemporal change detection
STAGE1_PATCH_SIZE = 2048
STAGE1_CHANGE_QUANTILE = 0.9
STAGE1_SAM_PARAMS = {
    "pred_iou_thresh": 0.6,
    "stability_score_thresh": 0.4,
    "box_nms_thresh": 0.4,
    "crop_nms_thresh": 0.4,
    "min_mask_region_area": 10000,
}

# Stage 2: anomaly change detection
STAGE2_CHANGE_MAP_QUANTILE = 0.7
STAGE2_MIN_MASK_PIXELS = 1500
STAGE2_SAM_PARAMS = {
    "pred_iou_thresh": 0.5,
    "stability_score_thresh": 0.6,
    "min_mask_region_area": 1500,
}

# Runtime
DEFAULT_DEVICE = "cuda:0"

# Evaluation
EVAL_QUANTILE_THRESH = 0.945
EVAL_AREA_RATIO = 0.0003
EVAL_BACKGROUND_WEIGHT = 0.1
EVAL_MAX_EVENTS = 80
EVAL_NUM_CATEGORIES = 6
