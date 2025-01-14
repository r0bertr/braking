from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import yaml


@dataclass
class Config:
    path_to_exp_dir: Path
    path_to_db: Path
    columns: List[str]
    seq_names: List[str] = field(default_factory=lambda: [])
    method: Literal["accel", "gbdt"] = "gbdt"
    accel_threshold: float = -0.25
    gt_flags: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    offset: int = 0
    test_size: float = 0.2
    depth: int = 10
    n_iters: int = 10000
    lr: float = 0.001
    early_stopping_rounds: int = 1000
    random_state: int = 233
    verbose: bool = True

    def to_yaml(self, path: Path):
        yaml.safe_dump(self.__dict__, path)
    
    @staticmethod
    def from_yaml(path: Path) -> "Config":
        with open(path, "r") as f:
            conf = yaml.safe_load(f)
        return Config(**conf)
