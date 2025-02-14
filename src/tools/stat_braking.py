import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.braking import BRAKING_CAUSES, DATETIME_FMT_SEQ_NAME
from src.braking.io import img_files_to_mp4
from src.braking.utils import dts_to_frames, get_start_dt
from src.tools.utils import parse_args


@dataclass
class _Args():
    path_to_data: Path = "../data/kyushu_driving_database"
    path_to_db: Path = "../data/kyushu_driving_database/databases/cleaned/probe_data_annotated_distance.db"
    path_to_output: Path = "../data/kyushu_driving_database/experiments/stat_braking_flags"

if __name__ == "__main__":
    args: _Args = parse_args(_Args)

    args.path_to_output.mkdir(parents=True, exist_ok=True)

    braking_causes = BRAKING_CAUSES
    for i, cause in enumerate(braking_causes):
        if i == 0:
            continue

        sql = f"""
            SELECT
                csv_identifier,
                seq_name,
                datetime
            FROM
                probe_data
            WHERE
                braking_flag = {i}
            LIMIT 5;
        """

        with sqlite3.connect(args.path_to_db) as conn:
            df = pd.read_sql(sql, conn)
        
        for j, row in (pbar := tqdm(list(df.iterrows()))):
            pbar.set_description(f"Sampling cause {cause}: ")
            csv_id = row["csv_identifier"]
            seq_name = row["seq_name"]
            dt = row["datetime"]

            start_dt = get_start_dt(args.path_to_db, csv_id, seq_name)
            dt = pd.to_datetime(dt)
            frames, n_frames = dts_to_frames(
                args.path_to_data,
                csv_id,
                seq_name,
                pd.Series([dt]),
                start_dt,
            )
            frame = frames[0]
            frame_start = int(round(frame - 25))
            frame_end = int(round(frame + 25))
            path_to_img_dir = Path(args.path_to_data) / "images" / csv_id / seq_name
            path_to_images = sorted(path_to_img_dir.glob("*.png"))[frame_start:frame_end]
            dt_str = datetime.strftime(dt, DATETIME_FMT_SEQ_NAME)
            mp4_name = f"{cause.replace(' ', '_')}_{csv_id}_{seq_name}_{dt_str}.mp4"
            path_to_mp4 = Path(args.path_to_output) / mp4_name
            if not path_to_mp4.exists():
                img_files_to_mp4(
                    path_to_images,
                    path_to_mp4,
                )
            frame_int = int(round(frame))
            path_to_frame = path_to_img_dir / f"{frame_int:06d}.png"
            img_name = f"{cause.replace(' ', '_')}_{csv_id}_{seq_name}_{dt_str}_{frame_int:06d}.png"
            path_to_img = Path(args.path_to_output) / img_name
            if not path_to_img.exists():
                shutil.copy(path_to_frame, path_to_img)
