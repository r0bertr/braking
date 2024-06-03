import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from braking import DATETIME_FMT_SEQ_NAME


def seq_name_to_dt(seq_name: str) -> datetime:
    dt_str = seq_name[:15]
    return datetime.strptime(dt_str, DATETIME_FMT_SEQ_NAME)

def get_start_dt(path_to_db: Path, csv_id: str, seq_name: str) -> datetime:
    sql = f"SELECT MIN(datetime) FROM probe_data WHERE csv_identifier = '{csv_id}' AND seq_name = '{seq_name}';"
    with sqlite3.connect(str(path_to_db)) as conn:
        df = pd.read_sql(sql, conn)
    df["MIN(datetime)"] = pd.to_datetime(df["MIN(datetime)"])
    return df["MIN(datetime)"][0]

def dts_to_frames(path_to_data: Path, csv_id: str, seq_name: str, dts: pd.Series, start_dt: datetime) -> Tuple[pd.Series, int]:
    path_to_video = path_to_data / "videos" / csv_id / f"{seq_name}.avi"
    cap = cv2.VideoCapture(str(path_to_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path_to_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = (dts - start_dt).dt.total_seconds() * fps
    frames = frames.to_numpy()
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frames, n_frames

def draw_text(img, text,
    pos=(0, 0),
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=3,
    text_color=(0, 255, 0),
    font_thickness=2,
    text_color_bg=(0, 0, 0)
):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def shift_df(df: pd.DataFrame, offset: int) -> pd.DataFrame:
    df = df.copy(True)
    seqs_start_inds = df[["csv_identifier", "seq_name"]].drop_duplicates().index.values[1:]
    df[["braking_flag"]] = df[["braking_flag"]].shift(-offset)
    drop_inds = np.array([np.arange(start, stop) for start, stop in zip(seqs_start_inds - offset, seqs_start_inds)])
    drop_inds = drop_inds.reshape(-1)
    df.loc[drop_inds] = None
    df = df.dropna().reset_index(drop=True)
    return df
