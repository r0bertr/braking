import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.braking import COLUMN_GROUPS, DATETIME_FMT_SEQ_NAME
from src.braking.braking_detector import BrakingDetector
from src.braking.config import Config
from src.braking.io import load_seqs_with_braking
from src.braking.object_detector import ObjectDetector
from src.braking.plot import plot_boxplot
from src.braking.utils import dts_to_frames, get_start_dt, shift_df
from src.tools.utils import parse_args

_detector = ObjectDetector(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    device="cuda"
)

@dataclass
class _Args:
    path_to_cfgs: list = field(default_factory=lambda: [
        # "configs/sg20/gbdt_probe_pred_1sec.yaml",
        # "configs/sg20/gbdt_shadow_pred_1sec.yaml",
        # "configs/sg20/gbdt_dist_pred_1sec.yaml",
        "configs/full_data/gbdt_probe_pred_1secs.yaml",
        "configs/full_data/gbdt_dist_pred_1secs.yaml",
    ])
    path_to_images: Path = "../data/kyushu_driving_database/images"
    path_to_output: Path = "../data/kyushu_driving_database/experiments/distance/compare"
    seed: int = 233 

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
    fn1 = (y1 < 0.5) & (y == 1)
    fn2 = (y2 < 0.5) & (y == 1)
    tn1 = (y1 < 0.5) & (y == 0)
    tn2 = (y2 < 0.5) & (y == 0)
    tp1 = (y1 >= 0.5) & (y == 1)
    tp2 = (y2 >= 0.5) & (y == 1)

    fp1_and_tn2 = fp1 & tn2
    fp2_and_tn1 = fp2 & tn1
    fn1_and_tp2 = fn1 & tp2
    fn2_and_tp1 = fn2 & tp1

    return {
        "fp1_and_tn2": df[fp1_and_tn2],
        "fp2_and_tn1": df[fp2_and_tn1],
        "fn1_and_tp2": df[fn1_and_tp2],
        "fn2_and_tp1": df[fn2_and_tp1],
    }

def _create_video(path_to_out: Path, path_to_images: Path,
    frame_start: int, frame_end: int, fps: int = 10):
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

def _image_od(img_np: np.ndarray, detector: ObjectDetector) -> np.ndarray:
    img = torch.tensor(img_np.transpose(2, 0, 1), device="cuda")
    with torch.no_grad():
        outputs = detector.forward([{"image": img}])[0]
    scores = outputs["instances"].scores.cpu().numpy()
    classes = outputs["instances"].pred_classes.cpu().numpy()
    keep = (scores >= 0.5) & np.isin(classes, [1, 2, 3, 5, 6, 7])
    scores = scores[keep]
    n_dets = len(scores)
    if n_dets == 0:
        return img_np
    classes = classes[keep]
    box2ds = outputs["instances"].pred_boxes.tensor.cpu().numpy()[keep]
    masks = outputs["instances"].pred_masks.cpu().numpy()[keep]
    mask = np.zeros_like(masks[0], dtype=int)
    for i in range(len(masks)):
        mask[masks[i]] = i + 1
    return detector.draw_dets(img_np, box2ds, mask, classes, scores)


def _create_video_object_detection(path_to_out: Path, path_to_images: Path,
    frame_start: int, frame_end: int, fps: int = 10):
    image_files = sorted(path_to_images.glob('*.png'))[frame_start:frame_end]

    if len(image_files) == 0:
        print(f"No images found in {path_to_images}")
        return

    # Read the first image to get the frame size
    frame = cv2.imread(str(image_files[0]))
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video = cv2.VideoWriter(str(path_to_out), fourcc, fps, (width, height))

    for image_file in image_files:
        img_np = np.array(Image.open(image_file))
        img_np = _image_od(img_np, _detector)
        video.write(img_np[..., ::-1])

    # Release the video writer
    video.release()

def _plot_boxplot(col: str, pair_dict: dict, path_to_output: Path,
    path_to_db: Path):
    sql = f"""
        SELECT
            {col}
        FROM
            probe_data
        WHERE
            {{}};
    """
    with sqlite3.connect(path_to_db) as conn:
        df_braking = pd.read_sql(sql.format("braking_flag > 0"), conn)
        df_not_braking = pd.read_sql(sql.format("braking_flag = 0"), conn)
    
    labels = ["Postive", "Negative", "FP(P) & TN(D)", "FN(P) & TP(D)"]
    plot_data = [df_braking[col], df_not_braking[col], None, None]
    for key, df_selected in pair_dict.items():
        if key == "fp1_and_tn2":
            plot_data[2] = df_selected[col]
        elif key == "fn1_and_tp2":
            plot_data[3] = df_selected[col]
    plot_boxplot(plot_data, path_to_output=path_to_output,
        labels=labels, only_inliers=True)


def _main():
    args: _Args = parse_args(_Args)

    # Exp names, cfgs, and path to exps
    exp_names = [Path(p).stem for p in args.path_to_cfgs]
    cfgs: List[Config] = [Config.from_yaml(p) for p in args.path_to_cfgs]
    path_to_exps = [cfg.path_to_exp_dir / exp_name for (cfg, exp_name) in zip(cfgs, exp_names)]

    # Get offset
    offset = cfgs[0].offset
    for cfg in cfgs[1:]:
        assert cfg.offset == offset, "Offsets must be the same"
    
    # Assert same sequences
    seq_names = cfgs[0].seq_names
    for cfg in cfgs[1:]:
        assert cfg.seq_names == seq_names, "Sequence names must be the same"

    # Load models
    detectors = []
    for cfg, path_to_exp in zip(cfgs, path_to_exps):
        detector = BrakingDetector(cfg, path_to_model=path_to_exp / "model.bin")
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
            pair_dicts[(i, j)] = _analyze_pair(df_shifted_val,
                y_val, y_preds[i], y_preds[j])

    # Statistics and create videos
    for (i, j), pair_dict in pair_dicts.items():
        _plot_boxplot("accel", pair_dict,
            args.path_to_output / "boxplot_accel.pdf", cfgs[0].path_to_db)
        _plot_boxplot("accel_x", pair_dict,
            args.path_to_output / "boxplot_accel_x.pdf", cfgs[0].path_to_db)
        _plot_boxplot("accel_y", pair_dict,
            args.path_to_output / "boxplot_accel_y.pdf", cfgs[0].path_to_db)
        _plot_boxplot("accel_z", pair_dict,
            args.path_to_output / "boxplot_accel_z.pdf", cfgs[0].path_to_db)

        for key, df_selected in pair_dict.items():
            path_to_pair = args.path_to_output / f"{exp_names[i]}_{exp_names[j]}"
            path_to_pair.mkdir(parents=True, exist_ok=True)
            
            for col_group, cols in COLUMN_GROUPS.items():
                if not all(col in df_selected.columns for col in cols):
                    continue
                plot_data = [df_selected[col] for col in cols]
                path_to_boxplot = path_to_pair / f"{key}_{col_group}.png"
                plot_boxplot(
                    plot_data,
                    path_to_boxplot,
                    labels=cols,
                    title=f"{key} - {col_group}",
                    y_label=f"Values - {len(df_selected)}",
                )

            path_to_video_dir = path_to_pair / f"{key}"
            path_to_video_dir.mkdir(parents=True, exist_ok=True)
            sampled_rows = df_selected
            if len(df_selected) > 5:
                sampled_rows = df_selected.sample(n=5, random_state=args.seed)
            for _, row in (pbar := tqdm(list(sampled_rows.iterrows()))):
                pbar.set_description(f"Creating videos for {exp_names[i]}|{exp_names[j]} - {key}")
                csv_id = row.csv_identifier
                seq_name = row.seq_name
                dt = pd.to_datetime(row.datetime)
                dt_start = get_start_dt(cfgs[i].path_to_db, csv_id, seq_name)
                frames, _ = dts_to_frames(
                    args.path_to_images.parent,
                    csv_id,
                    seq_name,
                    pd.Series([dt]),
                    dt_start,
                )
                frame_start = int(round(frames[0] - 25))
                frame_end = frame_start + 50
                path_to_images = args.path_to_images / csv_id / seq_name
                path_to_out_video = path_to_video_dir / f"{csv_id}_{seq_name}_{dt.strftime(DATETIME_FMT_SEQ_NAME)}.mp4"
                path_to_out_video_od = path_to_video_dir / f"{csv_id}_{seq_name}_{dt.strftime(DATETIME_FMT_SEQ_NAME)}_od.mp4"
                if not path_to_out_video.exists():
                    _create_video(path_to_out_video, path_to_images, frame_start, frame_end)
                if not path_to_out_video_od.exists():
                    _create_video_object_detection(path_to_out_video_od, path_to_images, frame_start, frame_end)

                path_to_out_img = path_to_video_dir / f"{csv_id}_{seq_name}_{dt.strftime(DATETIME_FMT_SEQ_NAME)}"
                path_to_out_img.mkdir(parents=True, exist_ok=True)
                path_to_out_img_od = path_to_video_dir / f"{csv_id}_{seq_name}_{dt.strftime(DATETIME_FMT_SEQ_NAME)}_od"
                path_to_out_img_od.mkdir(parents=True, exist_ok=True)
                for _, path_to_image in enumerate(sorted(path_to_images.glob('*.png'))[frame_start:frame_end]):
                    path_to_image_out = path_to_out_img / path_to_image.name
                    path_to_image_out_od = path_to_out_img_od / path_to_image.name
                    if not path_to_image_out.exists():
                        shutil.copy(path_to_image, path_to_image_out)
                    if not path_to_image_out_od.exists():
                        img = np.array(Image.open(path_to_image))
                        Image.fromarray(_image_od(img, _detector)).save(path_to_image_out_od)


if __name__ == "__main__":
    _main()
