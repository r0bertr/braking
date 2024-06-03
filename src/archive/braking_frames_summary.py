import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


@dataclass
class Args(argparse.Namespace):
    path_to_data: Path = Path("/home/user/data/ITS/kyushu_driving_database")
    path_to_output: Path = Path("data")

    @property
    def path_to_braking_labels(self) -> Path:
        return self.path_to_data / "braking_frames_vsa_bts"

def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    args = Args()
    return parser.parse_args(namespace=args)

if __name__ == "__main__":
    args = parse_args()
    
    n_brakings = []
    csv_ids = [path.stem for path in args.path_to_braking_labels.glob("*")]
    for csv_id in (pbar := tqdm(csv_ids)):
        path_to_csv_id = args.path_to_braking_labels / csv_id
        seq_names = [path.stem for path in path_to_csv_id.glob("*")]
        for seq_name in seq_names:
            if seq_name[0] == ".":
                continue
            pbar.set_description(f"{csv_id}/{seq_name}")
            path_to_anno = path_to_csv_id / seq_name / "braking_frames.txt"
            path_to_images = args.path_to_data / "images" / csv_id / seq_name
            n_frames = len(list(path_to_images.glob("*.png")))
            if path_to_anno.exists():
                with open(path_to_anno) as fp:
                    lines = fp.readlines()
                braking_frames = [int(line.strip()) for line in lines]
                n_brakings.append((csv_id, seq_name, len(braking_frames), braking_frames, n_frames))
            else:
                n_brakings.append((csv_id, seq_name, 0, [], n_frames))

    with open(args.path_to_output / "n_brakings.csv", "w", newline="") as fp:
        csv_writer = csv.writer(fp)
        csv_writer.writerow(["csv_id", "seq_name", "n_brakings", "braking_frames", "n_frames"])
        csv_writer.writerows(n_brakings)
