from dataclasses import MISSING, dataclass
from pathlib import Path

import cv2
import detectron2.modeling
import fvcore.common.checkpoint
import hydra.core.global_hydra
import numpy as np
import torch
import tridet.modeling.dd3d
import tridet.utils.setup
from hydra import compose, initialize
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.braking import DATA_ROOT
from src.braking.io import (img_stream_to_mp4, load_seq_names_with_braking,
                            save_json)
from src.tools.utils import parse_args


@dataclass
class Args:
    path_to_output_dir: Path = MISSING
    path_to_model_ckpt: Path = Path(
        "/home/user/data/ITS/models/dd3d-v99-final.pth")
    device: str = "cuda"
    save_imgs: bool = False
    save_videos: bool = False
    batch_size: int = 20
    num_workers: int = 0
    score_threshold: float = 0.5
    debug: bool = False

class KyushuDataset(torch.utils.data.Dataset):
    def __init__(self, csv_id, video_name, device):
        self.device = device
        self.path_to_imgs = sorted(list(Path(DATA_ROOT / "images" / csv_id / video_name).glob("*.png")))

    def __len__(self):
        return len(self.path_to_imgs)

    def __getitem__(self, idx):
        path_to_img = self.path_to_imgs[idx]
        img = torch.tensor(np.array(Image.open(path_to_img)).transpose(2, 0, 1), device=self.device)
        data_dict = {
            "frame_id": idx,
            "intrinsics": torch.tensor(K, device=self.device),
            "image": img,
        }
        return data_dict

def draw_dets(img: np.ndarray, det: dict, K: np.ndarray) -> np.ndarray:
    img = np.array(img)
    for det_idx in range(det["corners"].shape[0]):
        corners = det["corners"][det_idx]
        score = det["score"][det_idx]
        lines = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)]
        color = tuple((255 * np.array(plt.cm.jet(det_idx % 10 / 10)[:3])))
        corners_proj = ((K @ corners[..., None])[..., 0] / corners[..., 2:3])[..., :2]
        left_top = corners_proj[0]
        for line in lines:
            pts_from = corners_proj[line[0]].astype(np.int32)
            pts_to = corners_proj[line[1]].astype(np.int32)
            thickness = 1
            if line in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                thickness = 3
            cv2.line(img, pts_from, pts_to, color=color, thickness=thickness)
        # box2d = det["box2d"][det_idx].astype(int)
        # cv2.rectangle(img, (box2d[0], box2d[1]), (box2d[2], box2d[3]), color=color, thickness=3)
        cv2.putText(img, f"{det_idx}:{score:.2f}", (int(left_top[0]), int(left_top[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img

def compute_mean_std(dataloader: DataLoader):
    means = []
    for data_dicts in (pbar := tqdm(dataloader)):
        pbar.set_description("Computing the mean of images")
        imgs = torch.stack([data_dict["image"].float() for data_dict in data_dicts]) # (B, C, H, W)
        means.append(torch.mean(imgs, dim=(2, 3)))
    mean = torch.mean(torch.cat(means), dim=0)

    variances = []
    for data_dicts in (pbar := tqdm(dataloader)):
        pbar.set_description("Computing the variance of images")
        imgs = torch.stack([data_dict["image"].float() for data_dict in data_dicts]) # (B, C, H, W)
        variances.append(torch.mean((imgs - mean[None, :, None, None])**2, dim=(2, 3)))
    std = torch.sqrt(torch.mean(torch.cat(variances), dim=0))
    return mean.view(-1, 1, 1), std.view(-1, 1, 1)

def collate_fn(x):
    return x

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def main():
    args: Args = parse_args(Args)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(config_path="../lib/dd3d/configs"):
        cfg = compose(config_name="defaults",
            overrides=[
                "+experiments=dd3d_kitti_v99",
                f"MODEL.CKPT={args.path_to_model_ckpt}"
            ])
    tridet.utils.setup.setup(cfg)
    model = detectron2.modeling.build_model(cfg)
    ckpt = cfg.MODEL.CKPT
    fvcore.common.checkpoint.Checkpointer(model).load(ckpt)
    model.eval()

    path_to_db = DATA_ROOT / "databases/cleaned/probe_data_annotated.db"
    if args.debug:
        seq_names = ["20171213_09/20171213_135628_001_Camera1"]
        args.save_imgs = True
        args.save_videos = True
    else:
        seq_names = load_seq_names_with_braking(path_to_db)

    with torch.no_grad():
        for seq_idx, seq_name in enumerate(seq_names):
            csv_id, video_name = seq_name.split('/')
            (args.path_to_output_dir / csv_id).mkdir(parents=True, exist_ok=True)
            dataloader = DataLoader(
                KyushuDataset(csv_id, video_name, args.device),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
            )
            model.pixel_mean, model.pixel_std = compute_mean_std(dataloader)
            dets = []
            if args.save_videos:
                video_stream, video_close = img_stream_to_mp4(args.path_to_output_dir / csv_id / f"{video_name}.mp4", width=640, height=480)
            if args.save_imgs:
                (args.path_to_output_dir / csv_id / video_name).mkdir(parents=True, exist_ok=True)
            pbar = tqdm(total=len(dataloader), leave=True, desc=f"{seq_idx+1}/{len(seq_names)} {seq_name}")
            for data_dicts in dataloader:
                out_dicts = model(data_dicts)
                for img_idx, out_dict in enumerate(out_dicts):
                    instances = out_dict["instances"]
                    # scores = ((instances.scores + instances.scores_3d) / 2).detach().cpu().numpy() # (N, )
                    scores = instances.scores_3d.detach().cpu().numpy()
                    mask = scores >= args.score_threshold # (N, )
                    corners = instances.pred_boxes3d.corners.detach().cpu().numpy() # (N, 8, 3)
                    box2ds = instances.pred_boxes.tensor.detach().cpu().numpy() # (N, 4)
                    det = {
                        "frame_id": data_dicts[img_idx]["frame_id"],
                        "corners": corners[mask],
                        "box2d": box2ds[mask],
                        "score": scores[mask]
                    }
                    if args.save_imgs or args.save_videos:
                        img = data_dicts[img_idx]["image"].permute(1, 2, 0).detach().cpu().numpy()
                        img_vis = draw_dets(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), det, K)
                        if args.save_videos:
                            video_stream(img_vis)
                        if args.save_imgs:
                            cv2.imwrite(str(args.path_to_output_dir / csv_id / video_name / f"{data_dicts[img_idx]['frame_id']:06d}.png"), img_vis)
                    dets.append(det)
                pbar.update()
            save_json(dets, args.path_to_output_dir / csv_id / f"{video_name}.json")
            if args.save_videos:
                video_close()
            pbar.close()

if __name__ == "__main__":
    main()
