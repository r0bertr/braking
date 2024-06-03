from dataclasses import dataclass, field
from pathlib import Path
from tabnanny import verbose
from typing import List, Literal

import dacite
from omegaconf import OmegaConf


@dataclass
class DetectorConfig:
    method: str = "accel"
    gt_flags: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    offset: int = 0

@dataclass
class AccelConfig(DetectorConfig):
    threshold: float = -0.25

@dataclass
class GBDTConfig(DetectorConfig):
    columns: List[str] = field(default_factory=lambda: list())
    test_size: float = 0.2
    n_iters: int = 10000
    lr: float = 0.01
    depth: int = 6
    early_stopping_rounds: int = 1000
    random_state: int = 114514
    verbose: bool = False

@dataclass
class Config:
    path_to_db: Path
    detector: DetectorConfig

    def to_yaml(self, path: Path) -> str:
        OmegaConf.save(self, path)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        def detector_hook(x):
            if x["method"] == "accel":
                return AccelConfig(**x)
            if x["method"] == "gbdt":
                return GBDTConfig(**x)
            raise ValueError

        conf = OmegaConf.load(path)
        conf = dacite.from_dict(cls, conf, config=dacite.Config(type_hooks={
            Path: lambda x: Path(x),
            DetectorConfig: detector_hook
        }))
        return conf
