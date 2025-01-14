import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.braking import DATA_ROOT, DATETIME_FMT, NUMERIC_COLUMNS
from src.braking.io import load_all_seqs_names
from src.braking.utils import seq_name_to_dt
from src.tools.utils import parse_args


@dataclass
class _Args:
    path_to_output: Path = DATA_ROOT / "databases/cleaned"
    path_to_seqs: Path = None
    columns: str = "speed,accel_x,accel_y,accel_z,latitude,longitude,direction,csv_identifier"
    use_txt_annotations: bool = False

    @property
    def path_to_db(self) -> Path:
        return DATA_ROOT / "databases" / "raw_data_with_weather_index.db"

def _seq_to_df(start_dt: datetime, end_dt: datetime, columns: str,
    db_cnx: sqlite3.Connection, braking_frames: list = None) -> pd.DataFrame:
    
    start_dt_str, end_dt_str = start_dt.strftime(DATETIME_FMT), end_dt.strftime(DATETIME_FMT)

    # Check GPS dt
    dt_type = "camera_datetime"
    sql_dt = f"SELECT gps_datetime FROM probe_data WHERE csv_identifier = '{csv_id}' AND gps_datetime = '{start_dt_str}'"
    df_dt = pd.read_sql(sql_dt, db_cnx)
    if len(df_dt) == 1:
        dt_type = "gps_datetime"

    # Read data from sql
    sql = f"SELECT MIN({dt_type}) AS {dt_type},{columns} FROM probe_data WHERE csv_identifier = '{csv_id}' AND {dt_type} >= '{start_dt_str}' AND {dt_type} <= '{end_dt_str}' GROUP BY {dt_type}"
    df = pd.read_sql(sql, db_cnx)
    if df.empty:
        return df

    # Rename datetime
    df = df.rename(columns={dt_type: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], format=DATETIME_FMT)
    
    # Insert seq_name
    df.insert(columns.split(",").index("csv_identifier") + 2, "seq_name", [seq_name] * len(df))

    # Set types
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    # Correct acceleration
    ## Find static acceleration
    df_static = df[["datetime", "accel_x", "accel_y", "accel_z"]][df["speed"] == 0].reset_index(drop=True)
    if df_static.empty:
        accel_static = np.zeros(3)
    else:
        df_static_dts = df_static["datetime"]
        df_static_dt_deltas = df_static_dts.diff().dt.total_seconds().fillna(0).astype(int).rename("dt_delta")
        idx_ignore = df_static_dt_deltas.index[df_static_dt_deltas != 1]
        n = 5
        idx_ignore = np.unique(np.concatenate([np.arange(max(i-n,0), min(i+n+1, len(df_static))) for i in idx_ignore]))
        accel_static = np.array(df_static.drop(idx_ignore).mean())[1:].astype(float)

    ## Correct acceleration
    df["accel_x"] = df["accel_x"] - accel_static[0]
    df["accel_y"] = df["accel_y"] - accel_static[1]
    df["accel_z"] = df["accel_z"] - accel_static[2]

    # Load braking labels
    accel = np.linalg.norm(np.array(df[["accel_x", "accel_y", "accel_z"]]), axis=-1)
    df.insert(5, "accel", accel)
    df.insert(len(df.columns), "braking_flag", [0] * len(df))
    df.insert(len(df.columns), "braking_description", [""] * len(df))
    if braking_frames is not None:
        braking_secs = [int(braking_frame // fps) for braking_frame in braking_frames]
        df.loc[df["datetime"].isin([start_dt + timedelta(seconds=sec) for sec in braking_secs]), "braking_flag"] = 1

    return df

if __name__ == "__main__":
    args: _Args = parse_args(_Args)

    if args.path_to_seqs is not None:
        with open(args.path_to_seqs) as fp:
            reader = csv.DictReader(fp)
            seqs = list(reader)
    else:
        seqs = load_all_seqs_names(DATA_ROOT)

    db_cnx = sqlite3.connect(args.path_to_db)
    dfs = []
    for seq in (pbar := tqdm(seqs)):
        csv_id = seq["csv_id"]
        seq_name = seq["seq_name"]
        pbar.set_description(f"{csv_id}:{seq_name}")

        braking_frames = None
        if args.use_txt_annotations:
            path_to_braking_labels = DATA_ROOT / "braking_frames_vsa_bts" / csv_id / seq_name / "braking_frames.txt"
            braking_frames = np.loadtxt(path_to_braking_labels, dtype=int, ndmin=1)

        start_dt = seq_name_to_dt(seq_name)
        path_to_video = DATA_ROOT / "videos" / csv_id / f"{seq_name}.avi"
        cap = cv2.VideoCapture(str(path_to_video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(n_frames / fps)
        end_dt = start_dt + timedelta(seconds=duration)

        df = _seq_to_df(start_dt, end_dt, args.columns, db_cnx, braking_frames=braking_frames)

        if not df.empty:
            dfs.append(df)

    db_cnx.close()
    df = pd.concat(dfs, ignore_index=True)
    index_cols = ["csv_identifier", "seq_name"]
    df = df.set_index(index_cols)
    out_db = sqlite3.connect(args.path_to_output / "probe_data.db")
    df.to_sql("probe_data", out_db, if_exists="replace", index_label=index_cols)
