from pathlib import Path

import numpy as np

DATA_ROOT = Path("/home/user/data/ITS/kyushu_driving_database")
DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
DATETIME_FMT_SEQ_NAME = "%Y%m%d_%H%M%S"
NUMERIC_COLUMNS = ["speed", "accel_x", "accel_y", "accel_z", "latitude", "longitude", "direction"]
CAT_COLUMNS = ["datetime"]
BRAKING_CAUSES = [
    "Not brake",
    "Vehicle",
    "Signal",
    "Pedestrian",
    "Obstacle",
    "Abnormal",
    "Turn",
    "Change line",
    "Safety confirm",
    "Mistake",
    "Parking",
    "Other"
]
K = np.array([
    [590.7319123397027, 0.0, 320.0],
    [0.0, 614.51158047623, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
COLUMN_GROUPS = {
    "acceleration": ["accel", "accel_x", "accel_y", "accel_z"],
    "speed": ["speed"],
    "distance": ["delta_dist_mean_1secs", "delta_dist_mean_3secs", "dist_mean_1secs", "dist_mean_3secs", "n_dets_1secs", "n_dets_3secs"],
    "location": ["latitude", "longitude"],
    "direction": ["direction"],
    "illumination": ["sg_height", "sg_sharpness", "sg_amplitude"],
}
