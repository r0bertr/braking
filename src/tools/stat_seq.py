"""
Compute the statistics of braking sequences
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.braking import DATA_ROOT
from src.braking.io import imgs_to_mp4, load_df_by_csv_and_seq
from src.braking.plot import plot_df_seq_columns
from src.braking.utils import dts_to_frames, get_start_dt
from src.tools.utils import parse_args


@dataclass
class Args(argparse.Namespace):
    path_to_db: Path = DATA_ROOT / "databases/cleaned/annotation_sessions/accelx_0.25/probe_data.db"
    path_to_output: Path = DATA_ROOT / "experiments/stats"
    csv_id: str = "20171213_01"
    seq_name: str = "20171213_141836_001_Camera1"
    accel_threshold: float = None
    accel_x_threshold: float = -0.25
    clip_length: int = 100
    separate_videos: bool = True

if __name__ == "__main__":
    args: Args = parse_args(Args)
    path_to_output = args.path_to_output / f"{args.csv_id}_{args.seq_name}"
    path_to_output.mkdir(parents=True, exist_ok=True)

    df = load_df_by_csv_and_seq(args.path_to_db, csv_ids=[args.csv_id], seq_names=[args.seq_name])
    plot_df_seq_columns(df, path_to_output / f"{args.seq_name}.png", ["accel_x"], thresholds=[args.accel_threshold, args.accel_x_threshold] if args.accel_threshold is not None else [args.accel_x_threshold])

    accel_condition = (df["accel_x"] <= args.accel_x_threshold)
    if args.accel_threshold is not None:
        accel_condition = accel_condition | ((df["accel"] >= args.accel_threshold) & (df["accel_x"] < 0))
    accel_dts = df["datetime"][accel_condition]
    start_dt = get_start_dt(args.path_to_db, args.csv_id, args.seq_name)
    accel_frames, n_frames = dts_to_frames(DATA_ROOT, args.csv_id, args.seq_name, accel_dts, start_dt)
    paths_to_imgs = []
    n = args.clip_length // 2
    paths_to_imgs_list = sorted(list((DATA_ROOT / "images" / args.csv_id / args.seq_name).glob("*.png")))
    for frame in accel_frames:
        frame = int(frame)
        start_frame = np.maximum(frame - n, 0).item()
        end_frame = np.minimum(frame + n + 1, n_frames)
        paths_to_imgs.append(paths_to_imgs_list[start_frame:end_frame])

    if args.separate_videos:
        for i, paths in enumerate(tqdm(paths_to_imgs)):
            accels = np.array(df[["accel_x", "accel_y", "accel_z", "accel"]][accel_condition].iloc[i])
            texts = [f"{i};{j}/{len(paths)};({accels[0]:.2f}, {accels[1]:.2f}, {accels[2]:.2f}), {accels[3]:.2f}" for j in range(len(paths))]
            imgs_to_mp4(paths, path_to_output / f"braking_{i:03d}.mp4", texts)
    else:
        paths = sum(paths_to_imgs)
        texts = []
        for i in range(len(paths_to_imgs)):
            accels = np.array(df[["accel_x", "accel_y", "accel_z", "accel"]][accel_condition].iloc[i])
            texts += [f"{i};{j}/{len(paths)};({accels[0]:.2f}, {accels[1]:.2f}, {accels[2]:.2f}), {accels[3]:.2f}" for j in range(len(paths_to_imgs[i]))]
        imgs_to_mp4(paths, path_to_output / "braking.mp4", texts)
