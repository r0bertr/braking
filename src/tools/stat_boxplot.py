import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.braking import COLUMN_GROUPS
from src.braking.config import Config
from src.braking.plot import plot_boxplot
from src.tools.utils import parse_args


@dataclass
class _Args:
    path_to_db: Path = "../data/kyushu_driving_database/databases/cleaned/probe_data_annotated_distance_shadow.db"
    path_to_output: Path = "../data/kyushu_driving_database/experiments/stat_boxplot"

if __name__ == "__main__":
    args: _Args = parse_args(_Args)

    args.path_to_output.mkdir(parents=True, exist_ok=True)

    for col_group, cols in COLUMN_GROUPS.items():
        if col_group != "illumination":
            continue
        sql = f"""
            SELECT
                {','.join(cols)}
            FROM
                probe_data
            WHERE
                {{}};
        """

        if col_group == "illumination":
            seq_names = Config.from_yaml("./configs/sg20/gbdt_shadow.yaml").seq_names
            seq_names = [seq_name.split("/")[1] for seq_name in seq_names]
            sql = sql.format("(" + " OR ".join([f"seq_name = '{seq_name}'" for seq_name in seq_names]) + ") AND {}")

        with sqlite3.connect(args.path_to_db) as conn:
            df_braking = pd.read_sql(sql.format("braking_flag > 0"), conn)
            df_not_braking = pd.read_sql(sql.format("braking_flag = 0"), conn)

        plot_boxplot(
            [df_braking[col] for col in cols],
            path_to_output=args.path_to_output / f"braking_{col_group}.png",
            labels=cols,
            title=f"Braking {col_group} Dist.",
            whis=[10, 90] if col_group == "illumination" else None,
        )

        plot_boxplot(
            [df_not_braking[col] for col in cols],
            path_to_output=args.path_to_output / f"not_braking_{col_group}.png",
            labels=cols,
            title=f"Not Braking {col_group} Dist.",
            whis=[10, 90] if col_group == "illumination" else None,
        )
