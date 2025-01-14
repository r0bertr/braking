import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.tools.utils import parse_args

np.random.seed(233)

@dataclass
class Args:
    path_to_seq_names: Path = "../data/kyushu_driving_database/databases/cleaned/seq_names_shadow_round_2.csv"
    path_to_db: Path = "../data/kyushu_driving_database/databases/cleaned/probe_data_annotated_distance_shadow.db"
    path_to_shadow_exps: Path = "../data/shadow-sg/experiments/kyushu"

def check_and_create_new_columns(db_cnx) -> None:
    new_columns = ["sg_height", "sg_sharpness", "sg_amplitude"]
    default_vals = [0.0, 0.0, 0.0]

    for i, new_column in enumerate(new_columns):
        sql = f"ALTER TABLE probe_data ADD COLUMN {new_column} REAL default {default_vals[i]};"
        try:
            db_cnx.execute(sql)
        except Exception as e:
            print(e)

def load_seq_names(path_to_seq_names) -> List[str]:
    seq_names = []
    with open(path_to_seq_names, "r") as f:
        for line in f:
            line = line.strip()
            if "," in line:
                seq_names.append("/".join(line.split(",")[:2]))
    return seq_names

def load_sg_info(path_to_shadow_exps, seq_name, path_to_vis) -> dict:

    _, seq_name = seq_name.split("/")
    path_to_exp = path_to_shadow_exps / seq_name
    path_to_exp = list(path_to_exp.glob("*"))[0]
    print("Loading SG information:", path_to_exp)

    path_to_sg = path_to_exp / "eval/iter_020000/gaussians_opti.txt"
    sgs = np.loadtxt(path_to_sg)  # (N, 7)
    lobe_axis = sgs[:, :3] / np.linalg.norm(sgs[:, :3], axis=1, keepdims=True)  # (N, 3)
    sharpness = np.exp(-sgs[:, 3:4])  # (N, 1)
    amplitude = np.abs(sgs[:, 4:])  # (N, 3)

    upper_hemi_dir = np.array([[0.0, -1.0, 0.0]])
    upper_mask = np.sum(lobe_axis * upper_hemi_dir, axis=1) > 0

    sg_info = {
        "sg_height": {
            "mean": np.mean(lobe_axis[upper_mask, 1]),
            "std": np.std(lobe_axis[upper_mask, 1])
        },
        "sg_sharpness": {
            "mean": np.mean(sharpness[upper_mask]),
            "std": np.std(sharpness[upper_mask])
        },
        "sg_amplitude": {
            "mean": np.mean(np.linalg.norm(amplitude[upper_mask], axis=1)),
            "std": np.std(np.linalg.norm(amplitude[upper_mask], axis=1))
        }
    }

    # Visualize sg information
    envmap = sg_to_envmap(np.array([[0.0, sg_info["sg_height"]["mean"], 0.0]]),
        np.array([[sg_info["sg_sharpness"]["mean"]]]),
        np.array([[sg_info["sg_amplitude"]["mean"]] * 3]))
    Image.fromarray(envmap).save(path_to_vis / f"{seq_name}.png")
    envmap_orignal = sg_to_envmap(lobe_axis[upper_mask],
        sharpness[upper_mask], amplitude[upper_mask])
    Image.fromarray(envmap_orignal).save(path_to_vis / f"{seq_name}_original.png")

    return sg_info

def sg_to_envmap(lobe_axis, sharpness, amplitude):
    height, width = 256, 512
    theta, phi = np.meshgrid(np.linspace(-np.pi, np.pi, width),
        np.linspace(0, np.pi, height), indexing="xy")
    viewdirs = np.stack([np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1)
    
    envmap = amplitude * np.exp(sharpness * (
        np.sum(viewdirs[..., None, :] * lobe_axis, axis=-1, keepdims=True)-1.))
    envmap = np.sum(envmap, axis=-2)
    envmap = (envmap / np.max(envmap) * 255).astype(np.uint8)

    return envmap

def update_sg_info_to_db(db_cnx, seq_name, sg_info):
    csv_id, seq_name = seq_name.split("/")

    # Get all datetime
    sql = f"SELECT datetime FROM probe_data WHERE csv_identifier = '{csv_id}' AND seq_name = '{seq_name}';"
    dts = pd.read_sql(sql, db_cnx)
    assert not dts.duplicated().any(), f"Duplicated rows detected in {seq_name}"
    db_cursor = db_cnx.cursor()
    for dt_str in tqdm(dts["datetime"]):
        sg_height = np.random.normal(sg_info["sg_height"]["mean"], sg_info["sg_height"]["std"])
        sg_sharpness = np.random.normal(sg_info["sg_sharpness"]["mean"], sg_info["sg_sharpness"]["std"])
        sg_amplitude = np.random.normal(sg_info["sg_amplitude"]["mean"], sg_info["sg_amplitude"]["std"])
        sql = f"UPDATE probe_data SET sg_height = {sg_height}, sg_sharpness = {sg_sharpness}, sg_amplitude = {sg_amplitude} WHERE csv_identifier = '{csv_id}' AND seq_name = '{seq_name}' AND datetime = '{dt_str}';"
        db_cursor.execute(sql)
    db_cnx.commit()

if __name__ == "__main__":
    args: Args = parse_args(Args)

    path_to_vis = args.path_to_db.parent / "shadow_vis"
    path_to_vis.mkdir(parents=True, exist_ok=True)

    # Create db connection
    db_cnx = sqlite3.connect(args.path_to_db)

    # Create new columns (sg_height, sg_sharpness, sg_amplitude) if necessary
    check_and_create_new_columns(db_cnx)

    # Load seq names
    seq_names = load_seq_names(args.path_to_seq_names)

    for seq_name in seq_names:
        print(f"Processing {seq_name}...")
        
        # Load sg information
        sg_info = load_sg_info(args.path_to_shadow_exps, seq_name, path_to_vis)

        # Update database
        update_sg_info_to_db(db_cnx, seq_name, sg_info)

    db_cnx.close()
