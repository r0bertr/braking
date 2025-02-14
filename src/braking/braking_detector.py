
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.braking import CAT_COLUMNS
from src.braking.analysis import compute_metrics, compute_pr_curve
from src.braking.config import Config
from src.braking.io import save_json
from src.braking.plot import plot_model
from src.braking.utils import shift_df


class BrakingDetector:
    def __init__(self, cfg: Config, path_to_model: Path = None) -> None:
        self.cfg = cfg
        self.model = None
        if path_to_model is not None:
            self.model = catboost.CatBoostClassifier()
            self.model.load_model(path_to_model)

    def forward(self, df: pd.DataFrame) -> np.ndarray:
        if self.cfg.method == "accel":
            accel_x = df["accel_x"]
            flags = (accel_x <= self.cfg.threshold).to_numpy().astype(float)
            return flags

        if self.cfg.method == "gbdt":
            y = self.model.predict_proba(df)[:, 1]
            return y

    def split_df(self, df: pd.DataFrame):
        if self.cfg.offset != 0:
            df = shift_df(df, self.cfg.offset)
        x = df.drop("braking_flag", axis=1)[self.cfg.columns]
        y = np.isin(df["braking_flag"].to_numpy(), self.cfg.gt_flags).astype(int)
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state
        )
        return x_train, x_val, y_train, y_val

    def train_gbdt(self, path_to_output: Path, df: pd.DataFrame) -> catboost.CatBoostClassifier:
        cat_features = [i for i, col in enumerate(self.cfg.columns) if col in CAT_COLUMNS]

        x_train, x_val, y_train, y_val = self.split_df(df)

        n_train = x_train.shape[0]
        n_val = x_val.shape[0]
        n_train_pos = np.sum(y_train)
        n_val_pos = np.sum(y_val)

        self.model = catboost.CatBoostClassifier(
            iterations=self.cfg.n_iters,
            learning_rate=self.cfg.lr,
            depth=self.cfg.depth,
            random_seed=self.cfg.random_state,
            train_dir=path_to_output / "catboost",
            scale_pos_weight=(n_train - n_train_pos) / n_train_pos,
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            verbose=self.cfg.verbose,
            task_type="GPU",
        )

        self.model.fit(
            x_train, y_train,
            cat_features=cat_features,
            eval_set=[(x_val, y_val)],
        )

        print(f"Train: {n_train} ({n_train_pos} pos), val: {n_val} ({n_val_pos} pos)")
        print('Model is fitted: ' + str(self.model.is_fitted()))
        print('Model params:')
        print(self.model.get_params())

        self.model.save_model(path_to_output / "model.bin")

        eval_dict = self._eval(x_val, y_val)
        plot_model(eval_dict, self.cfg.columns, path_to_output)
        save_json(eval_dict, path_to_output / "eval.json")

    def _eval(self, x_val: pd.DataFrame, y_val: np.ndarray) -> dict:
        y_preds_probe = self.forward(x_val)
        y_preds_50 = np.where(y_preds_probe >= 0.5, 1, 0)
        metric_dict = compute_metrics(y_val, y_preds_50)
        precisions, recalls, thresholds = compute_pr_curve(y_val, y_preds_probe)
        feature_importance = self.model.get_feature_importance()
        return {
            "metric_dict_50": metric_dict,
            "precisions": precisions[:-1],
            "recalls": recalls[:-1],
            "thresholds": thresholds,
            "feature_importance": feature_importance
        }
