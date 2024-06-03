import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from cv2 import threshold
from tqdm import tqdm

from braking.analysis import compute_metrics, plot_column, plot_xy
from braking.config import Config
from braking.braking_detector import Detector
from braking.io import load_df, save_dicts_csv
from stat_seq import parse_args


@dataclass
class Args(argparse.Namespace):
    path_to_cfg: Path = None
    path_to_output: Path = None
    threshold_start: float = 0.0
    threshold_end: float = -0.5
    threshold_step: float = -0.01

if __name__ == "__main__":
    args: Args = parse_args(Args)

    cfg = Config.from_yaml(args.path_to_cfg)
    args.path_to_output.mkdir(parents=True, exist_ok=True)
    df = load_df(cfg.path_to_db)
    df["datetime"] = pd.to_datetime(df["datetime"])

    y_true = df["braking_flag"].to_numpy()

    detector = Detector(cfg.detector)
    thresholds = np.arange(args.threshold_start, args.threshold_end + args.threshold_step, args.threshold_step, dtype=float)
    metrics_dicts = []
    for threshold in (pbar := tqdm(thresholds)):
        pbar.set_description(f"Threshold: {threshold:.2f}")
        detector.cfg.threshold = threshold.item()
        y_preds_probe = detector.detect(df)
        y_preds = np.where(y_preds_probe >= 0.5, 1, 0)
        metrics_dict = compute_metrics(y_true, y_preds)
        metrics_dicts.append(metrics_dict)

    for i, m in enumerate(metrics_dicts):
        m["threshold"] = thresholds[i]
    save_dicts_csv(metrics_dicts, args.path_to_output / "metrics.csv")

    precisions = [m["precision"] for m in metrics_dicts]
    recalls = [m["recall"] for m in metrics_dicts]
    f1_scores = [m["f1_score"] for m in metrics_dicts]
    plot_xy(thresholds,
        f1_scores,
        args.path_to_output / "f1_scores.png",
        "Threshold vs. F1 Score",
        x_label="Threshold",
        y_label="F1 Score")
    plot_xy(recalls,
        precisions,
        args.path_to_output / "pr_graph.png",
        "Recall vs. Precision",
        annotations=[f"{t:.2f}" for t in thresholds],
        x_label="Recall",
        y_label="Precision")

    best_metrics_dict = max(metrics_dicts, key=lambda m: m["f1_score"])
    example_csv_id = "20171213_17"
    example_seq_name = "20171220_153035_001_Camera1"
    plot_column(
        df.loc[(df["csv_identifier"] == example_csv_id) & (df["seq_name"] == example_seq_name)].reset_index(drop=True),
        "accel_x",
        args.path_to_output / f"{example_csv_id}_{example_seq_name}_accel_x.png",
    )
