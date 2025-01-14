import hashlib
import io
import json
import os
import shutil
import sqlite3
from base64 import encodebytes
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import yaml
from flask import Flask, jsonify, render_template, request
from PIL import Image

from src.annotator import (CLIP_LENGTH, DATA_ROOT, PATH_TO_DB,
                           PATH_TO_SESSIONS, PATH_TO_TEMP)
from src.braking import DATETIME_FMT
from src.braking.io import imgs_to_mp4, seq_to_mp4
from src.braking.utils import dts_to_frames, get_start_dt

app = Flask(__name__)

def encode_img(path_to_img: Path):
    pil_img = Image.open(path_to_img, "r")
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format="PNG")
    encoded_img = encodebytes(byte_arr.getvalue()).decode("ascii")
    return encoded_img

@app.route("/ping", methods=["GET"])
def ping_pong():
    return jsonify("pong!")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/sessions", methods=["GET"])
def get_sessions():
    sessions = []
    for path_to_sess in PATH_TO_SESSIONS.iterdir():
        with open(path_to_sess / "config.yaml") as f:
            sessions.append(yaml.safe_load(f))
    return jsonify(sessions)

@app.route("/session", methods=["POST"])
def create_session():
    session_data = request.form
    if (PATH_TO_SESSIONS / session_data["name"]).exists():
        return jsonify({"msg": "Session already exists."}), 400

    try:
        threshold = float(session_data["threshold"])
    except ValueError as e:
        return jsonify({"msg": f"Invalid threshold: {e}."}), 400
    
    path_to_sess = PATH_TO_SESSIONS / session_data["name"]
    path_to_sess.mkdir()

    cfg = {
        "name": session_data["name"],
        "threshold": threshold,
        "use_accel_x": session_data["use_accel_x"] == "true",
    }
    with open(path_to_sess / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    shutil.copyfile(PATH_TO_DB, path_to_sess / "probe_data.db")

    with sqlite3.connect(path_to_sess / "probe_data.db") as db_cnx:
        sql_accel_condition = f"accel_x <= {threshold}" if cfg["use_accel_x"] else f"accel >= {threshold} AND accel_x < 0"
        sql_update = f"UPDATE probe_data SET braking_flag = 1 WHERE {sql_accel_condition};"
        db_cnx.execute(sql_update)
        sql = f"SELECT csv_identifier, seq_name, count(*) AS n_satisfied_accel_condition FROM probe_data WHERE braking_flag = 1 GROUP BY csv_identifier, seq_name;"
        df_seqs = pd.read_sql(sql, db_cnx)

    df_seqs.to_csv(path_to_sess / "sequences.csv")
    unfinished_seq_strs = (df_seqs["csv_identifier"] + "/" + df_seqs["seq_name"]).tolist()
    status = {
        "unfinished": unfinished_seq_strs,
        "finished": [],
    }
    with open(path_to_sess / "status.json", "w") as f:
        json.dump(status, f, indent=2)

    return jsonify({"msg": "Session created."})

@app.route("/session/<session_name>", methods=["GET"])
def get_session(session_name):
    path_to_sess = PATH_TO_SESSIONS / session_name
    if not path_to_sess.exists():
        return jsonify({"msg": "Session not found."}), 400
    
    with open(path_to_sess / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    with open(path_to_sess / "status.json") as f:
        status = json.load(f)

    return jsonify({
        "config": cfg,
        "status": status,
    })

@app.route("/sequence/<session_name>/<csv_id>/<seq_name>", methods=["GET"])
def get_sequence(session_name, csv_id, seq_name):
    path_to_sess = PATH_TO_SESSIONS / session_name
    with open(path_to_sess / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    threshold = cfg["threshold"]
    use_accel_x = cfg["use_accel_x"]

    with sqlite3.connect(PATH_TO_SESSIONS / session_name / "probe_data.db") as db_cnx:
        df = pd.read_sql(f"SELECT * FROM probe_data WHERE csv_identifier = '{csv_id}' AND seq_name = '{seq_name}';", db_cnx)
    df["datetime"] = pd.to_datetime(df["datetime"])

    if use_accel_x:
        accel_labels = df["accel_x"] <= threshold
    else:
        accel_labels = (df["accel"] >= threshold) & (df["accel_x"] < 0)
    accel_label_dts = df["datetime"][accel_labels]

    ret_data = {
        "datetimes": accel_label_dts.dt.strftime(DATETIME_FMT).to_list(),
        "accel": df["accel"][accel_labels].to_list(),
        "accel_x": df["accel_x"][accel_labels].to_list(),
        "accel_y": df["accel_y"][accel_labels].to_list(),
        "accel_z": df["accel_z"][accel_labels].to_list(),
        "labels": df["braking_flag"][accel_labels].to_list(),
        "braking_description": df["braking_description"][accel_labels].to_list(),
    }

    return jsonify(ret_data)

@app.route("/clip/<session_name>/<csv_id>/<seq_name>/<datetime_str>", methods=["GET"])
def get_clip(session_name, csv_id, seq_name, datetime_str):
    start_dt = get_start_dt(PATH_TO_SESSIONS / session_name / "probe_data.db", csv_id, seq_name)
    dt = datetime.strptime(datetime_str, DATETIME_FMT)
    frame, n_frames = dts_to_frames(DATA_ROOT, csv_id, seq_name, pd.Series([dt]), start_dt)
    frame = int(frame[0])
    n = CLIP_LENGTH // 2
    start_frame = np.maximum(frame - n, 0)
    end_frame = np.minimum(frame + n + 1, n_frames)
    paths_to_imgs = sorted(list((DATA_ROOT / "images" / csv_id / seq_name).glob("*.png")))
    path_to_mp4 = PATH_TO_TEMP / f"{hashlib.sha1(str(time()).encode('utf-8')).hexdigest()}.mp4"
    imgs_to_mp4(paths_to_imgs[start_frame:end_frame], path_to_mp4, texts=[f"{i}/{end_frame-start_frame}" for i in range(end_frame-start_frame)])
    encoded_mp4 = encodebytes(open(path_to_mp4, "rb").read()).decode("ascii")
    os.remove(path_to_mp4)

    return jsonify(encoded_mp4)

@app.route("/clip/<session_name>/<csv_id>/<seq_name>/<datetime_str>", methods=["PUT"])
def set_braking_flag(session_name, csv_id, seq_name, datetime_str):
    braking_flag = int(request.form["braking_flag"])
    braking_description = request.form["braking_description"]

    path_to_session = PATH_TO_SESSIONS / session_name
    with sqlite3.connect(path_to_session / "probe_data.db") as db_cnx:
        sql = f"UPDATE probe_data SET braking_flag = {braking_flag}, braking_description = '{braking_description}' WHERE csv_identifier = '{csv_id}' AND seq_name = '{seq_name}' AND datetime = '{datetime_str}';"
        db_cnx.execute(sql)

    return jsonify({"msg": "Braking flag updated."})

@app.route("/sequence/<session_name>/<csv_id>/<seq_name>", methods=["PUT"])
def set_finished(session_name, csv_id, seq_name):
    path_to_session = PATH_TO_SESSIONS / session_name
    with open(path_to_session / "status.json") as f:
        status = json.load(f)
    seq_str = f"{csv_id}/{seq_name}"

    if seq_str in status["unfinished"]:
        with open(path_to_session / "status.json", "w") as f:
            status["unfinished"].remove(seq_str)
            status["finished"].append(seq_str)
            json.dump(status, f, indent=2)
    
    return jsonify({"msg": "Sequence marked as finished."})
