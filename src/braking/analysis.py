from pathlib import Path
from typing import List, Tuple

import catboost
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    f1 = f1_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_mat.ravel()
    p, r = precision_score(y_true, y_pred), recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": acc,
        "f1_score": f1,
        "precision": p,
        "recall": r,
        "tn": tn.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        "tp": tp.item()
    }

def compute_pr_curve(y_true: np.ndarray, y_pred_probe: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return precision_recall_curve(y_true, y_pred_probe)
