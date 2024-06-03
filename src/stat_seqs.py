"""
Compute the statistics of braking sequences
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.pyplot import xlabel

from braking import BRAKING_CAUSES
from braking.io import save_ax
from utils import parse_args


@dataclass
class Args:
    path_to_db: Path = Path("/home/user/data/ITS/kyushu_driving_database/databases/cleaned/probe_data_annotated_distance.db")
    path_to_output: Path = Path("/home/user/data/ITS/kyushu_driving_database/experiments/stats")
    accel_x_threshold: float = -0.25

if __name__ == "__main__":
    args: Args = parse_args(Args)

    sql = f"""
SELECT csv_identifier, seq_name, count(*) AS n_records,
    count(case when accel_x <= {args.accel_x_threshold} then 1 else null end) AS n_satisfied_accel_x_condition,
    count(case when braking_flag > 0 then 1 else null end) AS n_brakings
FROM probe_data
GROUP BY csv_identifier, seq_name
"""
    # sql_to_csv(sql, args.path_to_output / f"count_accel_x_{args.accel_x_threshold}_brakings_by_seq.csv", args.path_to_db, index=True)

    sql = f"""
SELECT braking_flag, count(*) AS n_records
FROM probe_data
WHERE accel_x <= {args.accel_x_threshold}
GROUP BY braking_flag
"""

    with sqlite3.connect(args.path_to_db) as cnx:
        df = pd.read_sql(sql, cnx)
    df.insert(0, "braking_cause", BRAKING_CAUSES)
    df.set_index("braking_cause", inplace=True)
    df.to_csv(args.path_to_output / f"count_brakings_by_flag.csv", index=True)
    ax_bar = df.plot.bar(y="n_records", rot=45, figsize=(6.0, 6.0), legend=False, xlabel="")
    ax_bar.bar_label(ax_bar.containers[0])
    ax_bar.tick_params(axis='both', labelsize=13)
    save_ax(ax_bar, args.path_to_output / f"count_brakings_by_flag_bar.pdf")
