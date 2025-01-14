from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from src.braking.io import load_json
from src.braking.plot import plot_multi_feature_importance, plot_multi_xy

path_to_exp = Path("../data/kyushu_driving_database/experiments/shadow")

n_pairs = 3
paths_to_exps = [
    path_to_exp / "gbdt_probe",
    path_to_exp / "gbdt_probe_pred_1sec",
    path_to_exp / "gbdt_probe_pred_2sec",
    path_to_exp / "gbdt_shadow",
    path_to_exp / "gbdt_shadow_pred_1sec",
    path_to_exp / "gbdt_shadow_pred_2sec",
    path_to_exp / "gbdt_dist",
    path_to_exp / "gbdt_dist_pred_1sec",
    path_to_exp / "gbdt_dist_pred_2sec",
    path_to_exp / "gbdt_dist_shadow",
    path_to_exp / "gbdt_dist_shadow_pred_1sec",
    path_to_exp / "gbdt_dist_shadow_pred_2sec",
]
exp_names = [
    "Probe,\ndet.",
    "Probe,\npred. 1sec",
    "Probe,\npred. 2secs",
    "Shadow,\ndet.",
    "Shadow,\npred. 1sec",
    "Shadow,\npred. 2secs",
    "Dist,\ndet.",
    "Dist,\npred. 1sec",
    "Dist,\npred. 2secs",
    "Dist+Shadow,\ndet.",
    "Dist+Shadow,\npred. 1sec",
    "Dist+Shadow,\npred. 2secs",
]
feature_names_probe = ["Datetime", "Speed", "Accel x", "Accel y", "Accel z", "Accel", "Latitude", "Longitude", "Direction"]
feature_names_dist = ["Delta Dist. 3","Dist. 3","Dets 3","Dist. 1","Delta Dist. 1","Dets 1"]
feature_names_shadow = ["Shadow Height", "Shadow Sharpness", "Shadow Amplitude"]
feature_names = [feature_names_probe] * n_pairs + \
    [feature_names_probe + feature_names_shadow] * n_pairs + \
    [feature_names_probe + feature_names_dist] * n_pairs + \
    [feature_names_probe + feature_names_dist + feature_names_shadow] * n_pairs

metrics_dicts = [load_json(path_to_exp / "eval.json") for path_to_exp in paths_to_exps]
recalls = [np.array(metric["recalls"]) for metric in metrics_dicts]
precisions = [np.array(metric["precisions"]) for metric in metrics_dicts]

# Latex table
latex_headers = ["Experiment", "Acc.\\%$\\uparrow$", "F$\\uparrow$",
    "P$\\uparrow$", "R$\\uparrow$", "FPR$\\downarrow$", "TN$\\uparrow$",
    "FP$\\downarrow$", "FN$\\downarrow$", "TP$\\uparrow$",
    "Time (s)$\\downarrow$"]
latex = " & ".join(latex_headers) + " \\\\ \\hline\n"
for i in range(n_pairs):
    rows = []
    for j in range(len(exp_names) // n_pairs):
        metric_dict = metrics_dicts[i + j * n_pairs]["metric_dict_50"]
        rows.append([
            exp_names[i + j * n_pairs].replace("\n", " "),
            "{:.2f}".format(metric_dict["accuracy"] * 100),
            "{:.3f}".format(metric_dict["f1_score"]),
            "{:.3f}".format(metric_dict["precision"]),
            "{:.3f}".format(metric_dict["recall"]),
            "{:.3f}".format(metric_dict["fp"] / (metric_dict["fp"] + metric_dict["tn"])),
            "{}".format(metric_dict["tn"]),
            "{}".format(metric_dict["fp"]),
            "{}".format(metric_dict["fn"]),
            "{}".format(metric_dict["tp"]),
            "-"
        ])
    for col, col_name in enumerate(latex_headers):
        if "downarrow" in col_name:
            bold_idx = np.argmin([row[col] for row in rows])
        elif "uparrow" in col_name:
            bold_idx = np.argmax([row[col] for row in rows])
        else:
            bold_idx = None
        if bold_idx is not None:
            rows[bold_idx][col] = "\\textbf{" + rows[bold_idx][col] + "}"
    for row in rows[:-1]:
        latex += " & ".join(row) + " \\\\\n"
    latex += " & ".join(rows[-1]) + " \\\\ \\hline\n"
print(latex)

# PR-curve
plot_multi_xy(recalls, precisions, path_to_exp / "pr_curve.pdf",
    labels=[exp_name.replace("\n", " ") for exp_name in exp_names],
    grid=True, x_limits=(0, 1), y_limits=(0, 1), figsize=(13, 6))
y_limits = [(0.5, 0.8), (0, 0.5), (0, 0.1), (0, 0.1), (0, 0.1), (0, 0.1)]
for i in range(n_pairs):
    exp_labels = [exp_names[i + j * n_pairs] for j in range(len(exp_names) // n_pairs)]
    exp_labels = [exp_label.replace("\n", " ") for exp_label in exp_labels]
    rec = [recalls[i + j * n_pairs] for j in range(len(recalls) // n_pairs)]
    pre = [precisions[i + j * n_pairs] for j in range(len(precisions) // n_pairs)]
    plot_multi_xy(rec, pre, path_to_exp / f"pr_curve_{i}secs.pdf", labels=exp_labels, grid=True,
        x_limits=(0.8, 1), y_limits=(0.6, 1))

# Feature importance
merge_importances = {
    "Dist": ["Delta Dist. 3", "Dist. 3", "Dets 3", "Dist. 1", "Delta Dist. 1", "Dets 1"],
    "Shadow": ["Shadow Height", "Shadow Sharpness", "Shadow Amplitude"]
}
for i in range(n_pairs):
    imp_exp_names = []
    imp_feature_names = []
    imp_feature_importances = []
    for j in range(len(exp_names) // n_pairs):
        imp_exp_names.append(exp_names[i + j * n_pairs])
        cur_feature_names = deepcopy(feature_names[i + j * n_pairs])
        cur_feature_importances = deepcopy(metrics_dicts[i + j * n_pairs]["feature_importance"])
        for k, v in merge_importances.items():
            merged_importance = 0
            n_features = len(cur_feature_names)
            feature_idx = 0
            while feature_idx < n_features:
                if feature_idx < len(cur_feature_names) and cur_feature_names[feature_idx] in v:
                    merged_importance += cur_feature_importances[feature_idx]
                    cur_feature_names.remove(cur_feature_names[feature_idx])
                    cur_feature_importances.remove(cur_feature_importances[feature_idx])
                else:
                    feature_idx += 1
            cur_feature_names.append(k)
            cur_feature_importances.append(merged_importance)
        imp_feature_names.append(cur_feature_names)
        imp_feature_importances.append(cur_feature_importances)
    plot_multi_feature_importance(
        labels=imp_exp_names,
        path_to_output=path_to_exp / f"feature_importance_{i}.pdf",
        feature_names_list=imp_feature_names,
        feature_importances_list=imp_feature_importances,
    )

for i in range(n_pairs):
    n_methods = len(exp_names) // n_pairs
    for j in range(n_methods):
        for k in range(j + 1, n_methods):
            idx1 = i + j * n_pairs
            idx2 = i + k * n_pairs
            p1 = metrics_dicts[idx1]["precisions"]
            p2 = metrics_dicts[idx2]["precisions"]
            p_p = mannwhitneyu(p1, p2).pvalue
            r1 = metrics_dicts[idx1]["recalls"]
            r2 = metrics_dicts[idx2]["recalls"]
            r_p = mannwhitneyu(r1, r2).pvalue
            exp_name1 = exp_names[idx1].replace("\n", " ")
            exp_name2 = exp_names[idx2].replace("\n", " ")
            print(f"{exp_name1} vs {exp_name2}: p = {p_p}, r = {r_p}, avg={np.mean([p_p, r_p])}")
    print()
