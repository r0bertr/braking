from dataclasses import MISSING, dataclass
from pathlib import Path

from src.braking.braking_detector import BrakingDetector
from src.braking.config import Config
from src.braking.io import load_seqs_with_braking
from src.tools.utils import parse_args


@dataclass
class _Args:
    path_to_cfg: Path = MISSING

if __name__ == "__main__":
    args: _Args = parse_args(_Args)

    cfg = Config.from_yaml(args.path_to_cfg)
    path_to_output = cfg.path_to_exp_dir / args.path_to_cfg.stem
    path_to_output.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(path_to_output / "config.yaml")
    detector = BrakingDetector(cfg)

    df = load_seqs_with_braking(cfg.path_to_db)
    if len(cfg.seq_names) > 0:
        seq_names = [seq_name.split("/")[-1] for seq_name in cfg.seq_names]
        df = df[df["seq_name"].isin(seq_names)]
        df = df.reset_index(drop=True)
    detector.train_gbdt(path_to_output, df)
