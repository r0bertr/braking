from dataclasses import MISSING, dataclass
from pathlib import Path

from braking.braking_detector import BrakingDetector
from braking.config import Config
from braking.io import load_seqs_with_braking
from utils import parse_args


@dataclass
class Args:
    path_to_cfg: Path = MISSING
    path_to_output: Path = MISSING
    path_to_grid_search: Path = None

if __name__ == "__main__":
    args: Args = parse_args(Args)

    args.path_to_output.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_yaml(args.path_to_cfg)
    df = load_seqs_with_braking(cfg.path_to_db)
    cfg.to_yaml(args.path_to_output / "config.yaml")
    detector = BrakingDetector(cfg.detector)
    detector.train_gbdt(args.path_to_output, df)
