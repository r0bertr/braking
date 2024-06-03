from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn import metrics

from braking.io import load_json
from braking.plot import (plot_feature_importance,
                          plot_multi_feature_importance, plot_multi_xy)

path_to_exp = Path("/home/user/data/ITS/kyushu_driving_database/experiments")

paths_to_exps = [
    path_to_exp / "gbdt_probe",
    path_to_exp / "gbdt_probe_pred",
    path_to_exp / "gbdt_probe_pred_2secs",
    path_to_exp / "gbdt_probe_pred_3secs",
    path_to_exp / "gbdt_probe_pred_4secs",
    path_to_exp / "gbdt_probe_pred_5secs",
    path_to_exp / "gbdt_dist",
    path_to_exp / "gbdt_dist_pred_1secs",
    path_to_exp / "gbdt_dist_pred_2secs",
    path_to_exp / "gbdt_dist_pred_3secs",
    path_to_exp / "gbdt_dist_pred_4secs",
    path_to_exp / "gbdt_dist_pred_5secs",
]
exp_names = [
    "Probe,\ndet.",
    "Probe,\npred. 1sec",
    "Probe,\npred. 2secs",
    "Probe,\npred. 3secs",
    "Probe,\npred. 4secs",
    "Probe,\npred. 5secs",
    "Dist,\ndet.",
    "Dist,\npred. 1sec",
    "Dist,\npred. 2secs",
    "Dist,\npred. 3secs",
    "Dist,\npred. 4secs",
    "Dist,\npred. 5secs",
]
n_pairs = 6
feature_names_probe = ["Datetime", "Speed", "Accel x", "Accel y", "Accel z", "Accel", "Latitude", "Longitude", "Direction"]
feature_names_dist = ["Datetime", "Speed", "Accel x", "Accel y", "Accel z", "Accel", "Latitude", "Longitude", "Direction", "Delta Dist. 3","Dist. 3","Dets 3","Dist. 1","Delta Dist. 1","Dets 1"]
feature_names = [feature_names_probe] * n_pairs + [feature_names_dist] * n_pairs

metrics_dicts = [
    load_json(path_to_exp / "eval.json") for path_to_exp in paths_to_exps
]
recalls = [np.array(metric["recalls"]) for metric in metrics_dicts]
precisions = [np.array(metric["precisions"]) for metric in metrics_dicts]

plot_multi_xy(recalls, precisions, path_to_exp / "pr_curve.pdf",
    labels=[exp_name.replace("\n", " ") for exp_name in exp_names],
    grid=True, x_limits=(0, 1), y_limits=(0, 1), figsize=(13, 6))

y_limits = [(0.5, 0.8), (0, 0.5), (0, 0.1), (0, 0.1), (0, 0.1), (0, 0.1)]
for i, exp_name in enumerate(exp_names[:n_pairs]):
    exp_labels = [exp_names[i], exp_names[i+n_pairs]]
    exp_labels = [exp_label.replace("\n", " ") for exp_label in exp_labels]
    plot_multi_xy([recalls[i], recalls[i+n_pairs]], [precisions[i], precisions[i+n_pairs]],
        path_to_exp / f"pr_curve_{i}secs.pdf", labels=exp_labels, grid=True,
        x_limits=(0.8, 1), y_limits=y_limits[i])

# Feature importance
n_exp_per_graph = 3
merge_importances = ["Delta Dist. 3", "Dist. 3", "Dets 3", "Dist. 1", "Delta Dist. 1", "Dets 1"]
for i in range(n_pairs // n_exp_per_graph):
    imp_exp_names = []
    imp_feature_names = []
    imp_feature_importances = []
    for j in range(n_exp_per_graph):
        imp_exp_names.append(exp_names[i*n_exp_per_graph + j])
        imp_feature_names.append(feature_names[i*n_exp_per_graph + j])
        imp_feature_importances.append(metrics_dicts[i*n_exp_per_graph + j]["feature_importance"])
        imp_exp_names.append(exp_names[i*n_exp_per_graph + j + n_pairs])
        imp_feature_names.append(feature_names[i*n_exp_per_graph + j + n_pairs][:9] + ["Distance"])
        imp_feature_importances.append(metrics_dicts[i*n_exp_per_graph + j + n_pairs]["feature_importance"][:9] + \
            [sum(metrics_dicts[i*n_exp_per_graph + j + n_pairs]["feature_importance"][9:])])
    plot_multi_feature_importance(
        labels=imp_exp_names,
        path_to_output=path_to_exp / f"feature_importance_{i}.pdf",
        feature_names_list=imp_feature_names,
        feature_importances_list=imp_feature_importances,
        legend=i == 0,
    )

# U-test
t_tests = {
    "precision": {},
    "recall": {}
}
recalls = [np.array(metric["recalls"])[::(len(metric["recalls"]) // 10000)] for metric in metrics_dicts]
precisions = [np.array(metric["precisions"])[::(len(metric["recalls"]) // 10000)] for metric in metrics_dicts]
for i, exp_name in enumerate(exp_names[:n_pairs]):
    t_tests["precision"][exp_name] = mannwhitneyu(precisions[i], precisions[i+n_pairs])[1]
    t_tests["recall"][exp_name] = mannwhitneyu(recalls[i], recalls[i+n_pairs])[1]
print("t_test:\n", t_tests)
