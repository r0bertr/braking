import csv
import json
import os
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL import Image

from src.braking.utils import draw_text


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_df_by_csv_and_seq(path_to_db: Path, csv_ids = [], seq_names = []) -> pd.DataFrame:
    sql = f"SELECT * FROM probe_data"
    if len(csv_ids) > 0 or len(seq_names) > 0:
        sql += " WHERE "
        if len(csv_ids) == 0:
            if len(seq_names) > 1:
                sql += f"seq_name IN {tuple(seq_names)}"
            else:
                sql += f"seq_name = '{seq_names[0]}'"
        elif len(seq_names) == 0:
            if len(csv_ids) > 1:
                sql += f"csv_identifier IN {tuple(csv_ids)}"
            else:
                sql += f"csv_identifier = '{csv_ids[0]}'"
        else:
            if len(csv_ids) == 1:
                sql += f"csv_identifier = '{csv_ids[0]}' AND "
            else:
                sql += f"csv_identifier IN {tuple(csv_ids)} AND "
            if len(seq_names) == 1:
                sql += f"seq_name = '{seq_names[0]}'"
            else:
                sql += f"seq_name IN {tuple(seq_names)}"
    with sqlite3.connect(path_to_db) as conn:
        df = pd.read_sql(sql, conn)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def load_seqs_with_braking(path_to_db: Path, seq_names: list = None) -> pd.DataFrame:
    sql = f"""SELECT t1.*
FROM probe_data AS t1
JOIN (SELECT DISTINCT csv_identifier, seq_name
    FROM probe_data
    WHERE braking_flag > 0
    GROUP BY csv_identifier, seq_name) AS t2
ON t1.csv_identifier = t2.csv_identifier AND t1.seq_name = t2.seq_name
ORDER BY csv_identifier, seq_name, datetime;
""" # ORDER BY datetime is important. 
    # To reproduce the paper results, remove datetime after ORDER BY here.
    with sqlite3.connect(path_to_db) as conn:
        df = pd.read_sql(sql, conn)
    return df

def load_seq_names_with_braking(path_to_db: Path) -> list:
    with sqlite3.connect(path_to_db) as conn:
        sql = "SELECT DISTINCT csv_identifier,seq_name FROM probe_data WHERE braking_flag > 0 GROUP BY csv_identifier,seq_name ORDER BY csv_identifier,seq_name;"
        df = pd.read_sql(sql, conn)
    seq_names = (df["csv_identifier"] + "/" + df["seq_name"]).tolist()
    return seq_names

def save_dicts_csv(data: list, path_to_output: Path, float_fmt: str = ".4f"):
    with open(path_to_output, "w+") as fp:
        writer = csv.DictWriter(fp, fieldnames=data[0].keys())
        writer.writeheader()
        for d in data:
            writer.writerow({k: f"{v:{float_fmt}}" if isinstance(v, float) else v for k, v in d.items()})

def save_lists_csv(data: list, path_to_output: Path, float_fmt: str = ".4f"):
    with open(path_to_output, "w+") as fp:
        writer = csv.writer(fp)
        for d in data:
            writer.writerow([f"{v:{float_fmt}}" if isinstance(v, float) else v for v in d])

def save_json(data, path_to_output: Path, indent=2):
    with open(path_to_output, "w+") as fp:
        json.dump(data, fp, indent=indent, cls=NumpyEncoder)

def load_json(path_to_output: Path):
    with open(path_to_output, "r") as fp:
        return json.load(fp)

def load_all_seqs_names(path_to_data_root: Path) -> list:
    seqs = []
    for csv_id in (path_to_data_root / "videos").glob("*"):
        csv_id = csv_id.stem
        if len(csv_id) != 11:
            continue
        for seq_name in (path_to_data_root / "videos" / csv_id).glob("*.avi"):
            seq_name = seq_name.stem
            if seq_name[:4] == "2015":
                continue
            seqs.append({"csv_id": csv_id, "seq_name": seq_name})
    return seqs

def img_files_to_mp4(paths_to_imgs: list, path_to_output: Path, texts: list = None, fps: int = 10):
    img_size = Image.open(paths_to_imgs[0]).size
    path_to_tmp_output = path_to_output.parent / f"{path_to_output.stem}_.mp4"
    video = cv2.VideoWriter(str(path_to_tmp_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img_size[0], img_size[1]))
    for i, path in enumerate(paths_to_imgs):
        img = cv2.imread(str(path))
        if texts is not None:
            draw_text(img, texts[i], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        video.write(img)
    video.release()
    os.system(f"/usr/bin/ffmpeg -y -i {path_to_tmp_output} -pix_fmt yuv420p -vcodec libx264 {path_to_output} >/dev/null 2>&1")
    os.remove(path_to_tmp_output)

def img_stream_to_mp4(path_to_output: Path, width: int, height: int, fps: int = 10):
    path_to_tmp_output = path_to_output.parent / f"{path_to_output.stem}_.mp4"
    video = cv2.VideoWriter(str(path_to_tmp_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    def write(img: np.ndarray):
        video.write(img)

    def close():
        video.release()
        os.system(f"/usr/bin/ffmpeg -y -i {path_to_tmp_output} -pix_fmt yuv420p -vcodec libx264 {path_to_output} >/dev/null 2>&1")
        os.remove(path_to_tmp_output)

    return write, close

def sql_to_csv(sql: str, path_to_output: Path, path_to_db: Path, index: bool = False):
    with sqlite3.connect(path_to_db) as conn:
        df = pd.read_sql(sql, conn)
    df.to_csv(path_to_output, index=index)

def save_ax(ax: Axes, path_to_output: Path, tight: bool = True):
    fig = ax.get_figure()
    if tight:
        fig.tight_layout()
    fig.savefig(path_to_output)
    plt.close(fig)
