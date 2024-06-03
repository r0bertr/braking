from pathlib import Path
from turtle import color
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

MARKERS = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "P", "*"]

def plot_multi_feature_importance(labels, path_to_output: Path,
    feature_names_list, feature_importances_list,
    title: str = "", rot=30, figsize=(6.0, 8.22), tight=True,
    fontsize=13, legend: bool = True) -> None:
    fig, ax = plt.subplots()

    unified_feature_names = max(feature_names_list, key=len)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unified_feature_names)))
    importances = {}
    for feature_names, feature_importances in zip(feature_names_list, feature_importances_list):
        for i, feature_name in enumerate(unified_feature_names):
            if feature_name not in importances:
                importances[feature_name] = []
            if i < len(feature_names):
                importances[feature_name].append(feature_importances[i])
            else:
                importances[feature_name].append(0.0)

    bottom = np.zeros(len(labels))
    for i, (feature_name, feature_importances) in enumerate(importances.items()):
        feature_importances = np.array(feature_importances)
        ax.bar(labels, importances[feature_name], label=feature_name, bottom=bottom, color=colors[i])
        bottom += feature_importances
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.tick_params(axis="x", labelrotation=rot)
    if legend:
        ax.legend(bbox_to_anchor=(0.5, 1.24), loc="upper center", ncol=3, fancybox=True, shadow=True, fontsize=fontsize)
    ax.set_title(title)
    if figsize is not None:
        fig.set_size_inches(figsize)
    if tight:
        fig.tight_layout()
    fig.savefig(path_to_output)
    plt.close(fig)

def plot_feature_importance(feature_names, feature_importances, path_to_output: Path, title: str = "", rot: int = 45, tight: bool = True) -> None:
    fig, ax = plt.subplots()
    ax.bar(feature_names, feature_importances)
    ax.tick_params(axis="x", labelrotation=rot)
    ax.set_title(title)
    fig.set_size_inches(6.4, 6.8)
    if tight:
        fig.tight_layout()
    fig.savefig(path_to_output)
    plt.close(fig)

def plot_model(data: dict, feature_names: list, path_to_output: Path) -> None:
    p, r, th = data["precisions"], data["recalls"], data["thresholds"]
    plot_xy(
        p, r, path_to_output / "pr_curve.pdf",
        title="PR Curve",
        x_label="Recalls", y_label="Precisions",
    )
    plot_feature_importance(feature_names, data["feature_importance"], path_to_output / "feature_importance.pdf", title="Feature Importance")

def plot_df_seq_column(df: pd.DataFrame, column: str, path_to_output: Path) -> None:
    csv_id = df["csv_identifier"][0]
    seq_name = df["seq_name"][0]
    dt = df["datetime"]
    braking_flags = df["braking_flag"]
    braking_inds = braking_flags[braking_flags == 1].index
    x_data = (dt - dt[0]).dt.total_seconds()
    y_data = df[column]

    fig, ax = plt.subplots()
    ax.plot(x_data, y_data)
    ax.set_title(f"{csv_id}/{seq_name} - {column}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(column)
    for idx in braking_inds:
        ax.axvline(x=idx, color="red", linestyle="dashed", alpha=0.25)
    fig.savefig(path_to_output)
    plt.close(fig)

def plot_df_seq_columns(df: pd.DataFrame, path_to_output: Path, columns: List[str], thresholds: list = None) -> None:
    csv_id = df["csv_identifier"][0]
    seq_name = df["seq_name"][0]
    dt = df["datetime"]
    braking_flags = df["braking_flag"]
    braking_inds = braking_flags[braking_flags == 1].index
    x_data = (dt - dt[0]).dt.total_seconds()
    markers = MARKERS[:len(columns)]
 
    fig, ax = plt.subplots()
    for i, column in enumerate(columns):
        ax.plot(x_data, df[column], marker=markers[i], label=column)
    ax.set_title(f"{csv_id}/{seq_name}")
    ax.set_xlabel("Time (s)")
    ax.legend()
    for idx in braking_inds:
        ax.axvline(x=idx, color="red", linestyle="dashed", alpha=0.25)
    if thresholds is not None:
        for threshold in thresholds:
            ax.axhline(y=threshold, color="green", linestyle="dashed", alpha=0.25)
    fig.savefig(path_to_output)
    plt.close(fig)

def plot_xy(x_data, y_data, path_to_output: Path, title: str = "", x_label: str = "", y_label: str = "", annotations = None) -> None:
    fig, ax = plt.subplots()
    marker = MARKERS[0]
    ax.plot(x_data, y_data, marker=marker)
    if annotations is not None:
        for i, anno in enumerate(annotations):
            ax.annotate(anno, (x_data[i], y_data[i]))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig(path_to_output)
    plt.close(fig)

def plot_multi_xy(x_datas, y_datas, path_to_output: Path, title: str = "",
    x_label: str = "", y_label: str = "", x_limits: tuple = None, y_limits: tuple = None,
    markers: List[str] = None, labels: List[str] = [], annotations = None, grid = False,
    fontsize: int = 13, figsize=None, tight: bool = True) -> None:
    fig, ax = plt.subplots()
    if markers is None:
        markers = [None] * len(x_datas)
    if x_limits is not None:
        ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    for i, (x_data, y_data, marker) in enumerate(zip(x_datas, y_datas, markers)):
        if len(labels) > i:
            label = labels[i]
        else:
            label = None
        ax.plot(x_data, y_data, marker=marker, label=label)
        if annotations is not None:
            for j, anno in enumerate(annotations[i]):
                ax.annotate(anno, (x_data[j], y_data[j]))
    if len(labels) > 0:
        # ax.legend(fontsize=fontsize)
        ax.legend(bbox_to_anchor=(1.1, 1.0), loc="upper right", fontsize=fontsize)
    if grid:
        ax.grid()
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if figsize is not None:
        fig.set_size_inches(figsize)
    if tight:
        fig.tight_layout()
    fig.savefig(path_to_output)
    plt.close(fig)
