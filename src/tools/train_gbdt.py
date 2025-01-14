from dataclasses import MISSING, dataclass
from pathlib import Path

from src.braking.braking_detector import BrakingDetector
from src.braking.config import Config
from src.braking.io import load_seqs_with_braking
from src.tools import parse_args


@dataclass
class Args:
    path_to_cfg: Path = MISSING
    path_to_output: Path = MISSING

if __name__ == "__main__":
    args: Args = parse_args(Args)

    args.path_to_output.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_yaml(args.path_to_cfg)
    df = load_seqs_with_braking(cfg.path_to_db)
    if len(cfg.seq_names) > 0:
        seq_names = [seq_name.split("/")[-1] for seq_name in cfg.seq_names]
        df = df[df["seq_name"].isin(seq_names)]
        df = df.reset_index(drop=True)
    cfg.to_yaml(args.path_to_output / "config.yaml")
    detector = BrakingDetector(cfg.detector)
    detector.train_gbdt(args.path_to_output, df)