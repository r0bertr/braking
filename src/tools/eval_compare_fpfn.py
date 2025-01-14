from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from src.braking import DATETIME_FMT, DATETIME_FMT_SEQ_NAME
from src.braking.braking_detector import BrakingDetector
from src.braking.config import Config
from src.braking.io import load_seqs_with_braking
from src.braking.utils import shift_df
from src.tools import parse_args


@dataclass
class _Args:
    path_to_cfgs: list = field(default_factory=lambda: [
        "configs/shadow/gbdt_probe_pred_1sec.yaml",
        "configs/shadow/gbdt_shadow_pred_1sec.yaml",
        "configs/shadow/gbdt_dist_pred_1sec.yaml",
    ])
    path_to_exp_root: Path = "../data/kyushu_driving_database/experiments/shadow"
    path_to_images: Path = "../data/kyushu_driving_database/images"

def _load_df(cfg: Config) -> pd.DataFrame:
    df = load_seqs_with_braking(cfg.path_to_db)
    if len(cfg.seq_names) > 0:
        seq_names = [seq_name.split("/")[-1] for seq_name in cfg.seq_names]
        df = df[df["seq_name"].isin(seq_names)]
        df = df.reset_index(drop=True)
    return df

def _analyze_pair(df, y, y1, y2):
    fp1 = (y1 >= 0.5) & (y == 0)
    fp2 = (y2 >= 0.5) & (y == 0)
    # fn1 = (y1 < 0.5) & (y == 1)
    # fn2 = (y2 < 0.5) & (y == 1)
    tn1 = (y1 < 0.5) & (y == 0)
    tn2 = (y2 < 0.5) & (y == 0)
    # tp1 = (y1 >= 0.5) & (y == 1)
    # tp2 = (y2 >= 0.5) & (y == 1)

    fp1_and_tn2 = fp1 & tn2
    fp2_and_tn1 = fp2 & tn1
    fp1_and_fp2 = fp1 & fp2

    return {
        "fp1_and_tn2": df[fp1_and_tn2],
        "fp2_and_tn1": df[fp2_and_tn1],
        # "fp1_and_fp2": df[fp1_and_fp2],
    }

def _create_video(path_to_out: Path, path_to_images: Path,
    frame_start: int, frame_end: int, fps: int = 5):
    image_files = sorted(path_to_images.glob('*.png'))[frame_start:frame_end]

    if len(image_files) == 0:
        print(f"No images found in {path_to_images}")
        return

    # Read the first image to get the frame size
    frame = cv2.imread(str(image_files[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(str(path_to_out), fourcc, fps, (width, height))

    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        video.write(frame)

    # Release the video writer
    video.release()

def _main():
    args: _Args = parse_args(_Args)

    # Exp names, cfgs, and path to exps
    exp_names = [Path(p).stem for p in args.path_to_cfgs]
    cfgs = [Config.from_yaml(p) for p in args.path_to_cfgs]
    path_to_exps = [args.path_to_exp_root / exp_name for exp_name in exp_names]

    # Get offset
    offset = cfgs[0].detector.offset
    for cfg in cfgs[1:]:
        assert cfg.detector.offset == offset, "Offsets must be the same"
    
    # Assert same sequences
    seq_names = cfgs[0].seq_names
    for cfg in cfgs[1:]:
        assert cfg.seq_names == seq_names, "Sequence names must be the same"

    # Load models
    detectors = []
    for cfg, path_to_exp in zip(cfgs, path_to_exps):
        detector = BrakingDetector(cfg.detector,
            path_to_model=path_to_exp / "model.bin")
        detectors.append(detector)

    # Load data
    df = _load_df(cfgs[0])
    df_shifted = shift_df(df, offset)
    x_vals = [detector.split_df(df)[1] for detector in detectors]
    df_shifted_val = df_shifted.loc[x_vals[0].index]
    y_val = detectors[0].split_df(df)[3]

    # Eval
    y_preds = [detector.forward(x_val) for detector, x_val in zip(detectors, x_vals)]

    # Compare
    pair_dicts = {}
    for i in range(len(exp_names)):
        for j in range(i+1, len(exp_names)):
            print(f"Comparing {exp_names[i]} and {exp_names[j]}")
            pair_dicts[(i, j)] = _analyze_pair(df_shifted_val,
                y_val, y_preds[i], y_preds[j])

    path_to_out_dir = args.path_to_exp_root / "compare"
    for (i, j), pair_dict in pair_dicts.items():
        for key, df_selected in pair_dict.items():
            path_to_out = path_to_out_dir / f"{exp_names[i]}_{exp_names[j]}/{key}"
            path_to_out.mkdir(parents=True, exist_ok=True)
            print(f"Performing {path_to_out}")
            for _, row in df_selected.iterrows():
                csv_id = row.csv_identifier
                seq_name = row.seq_name
                dt = datetime.strptime(row.datetime, DATETIME_FMT)
                print(f"{csv_id}/{seq_name}/{row.datetime}:")
                print(f"  {row.speed}, {row.accel_x}, {row.accel}, {row.latitude}, {row.longitude}")
                dt_start = datetime.strptime(seq_name[:15], DATETIME_FMT_SEQ_NAME)
                frame_start = int((dt - dt_start).total_seconds() * 5) - 25
                frame_end = frame_start + 50
                path_to_images = args.path_to_images / csv_id / seq_name
                path_to_out_video = path_to_out / f"{csv_id}_{seq_name}_{dt.strftime(DATETIME_FMT_SEQ_NAME)}.mp4"
                _create_video(path_to_out_video, path_to_images, frame_start, frame_end)

if __name__ == "__main__":
    _main()
