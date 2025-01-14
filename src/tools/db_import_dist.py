import sqlite3
from argparse import Namespace
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from time import time

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
from bts.bts import BtsModel
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.braking import DATA_ROOT, DATETIME_FMT, K
from src.braking.dataset import KyushuDataset
from src.braking.io import load_seq_names_with_braking, save_json
from src.braking.utils import draw_text, dts_to_frames
from src.tools.utils import parse_args


@dataclass
class _Args:
    path_to_db: Path = MISSING
    detector_cfg_url: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    class_inds: str = "1,2,3,5,6,7"
    score_threshold: float = 0.5
    path_to_depth_ckpt: Path = Path("/home/user/data/ITS/models/bts_eigen_v2_pytorch_densenet161/model")
    seq_start_idx: int = 0
    seq_end_idx: int = -1
    batch_size: int = 60
    num_workers: int = 4
    save_dets: bool = False
    save_vis: bool = False
    device: str = "cuda"
    debug: bool = False

class _Detector:
    def __init__(self, cfg_url: str, device: str = "cuda") -> None:
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(cfg_url))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_url)
        self.model = build_model(self.cfg)
        DetectionCheckpointer(self.model).load(self.cfg.MODEL.WEIGHTS)
        self.model.eval()
        self.model.to(device)
    def forward(self, input_dicts: list) -> Tensor:
        outputs = self.model(input_dicts)
        return outputs

    @property
    def thing_classes(self) -> list:
        return MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes

class _DepthEstimator:
    def __init__(self, path_to_model: Path, focal: float, width: int, height: int,
        dataset: str = "kitti", encoder: str = "densenet161_bts",
        model_name: str = "bts_eigen_v2_pytorch_densenet161",
        bts_size: int = 512, max_depth: float = 80, training_focal: float = 715.0873,
        device: str = "cuda") -> None:

        self.params = Namespace(
            dataset=dataset,
            focal=focal,
            input_width=width,
            input_height=height,
            encoder=encoder,
            model_name=model_name,
            bts_size=bts_size,
            max_depth=max_depth,
            training_focal=training_focal
        )
        self.model = BtsModel(self.params)
        self.model = torch.nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(path_to_model)["model"])
        self.model.to(device)
        self.model.eval()
        self.device = device

    def forward(self, img: Tensor) -> Tensor:
        h, w = img.shape[-2:]
        if h % 32 != 0 or w % 32 != 0:
            img = torchvision.transforms.Resize((h // 32 * 32, w // 32 * 32))(img)
        img = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        focal = torch.tensor([self.params.focal], device=self.device)
        depth = self.model(img, focal)[-1]
        if h % 32 != 0 or w % 32 != 0:
            depth = torchvision.transforms.Resize((h, w))(depth)
        return depth

def _collate_fn(x: list):
    return [{"image": img} for img in x]
    
def _draw_dets(img: np.ndarray, box2d: np.ndarray, mask: np.ndarray, classes: np.ndarray, scores: np.ndarray, thing_classes: list) -> None:
    colors = plt.cm.Paired(np.arange(0, box2d.shape[0] + 1))[:, :3] # (N + 1, 3)
    colors[0] = 0
    colors = (colors * 255).astype(np.uint8)
    colored_mask = colors[mask]
    colored_mask[mask == 0] = img[mask == 0]
    img = (img.astype(np.float32) + colored_mask.astype(np.float32)) / 2
    img = img.astype(np.uint8)
    for i, bbox in enumerate(box2d):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        class_idx = classes[i]
        class_name = thing_classes[class_idx]
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[i+1].tolist(), 2)
        cv2.putText(img, f"{i+1}:{class_name}:{scores[i]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i+1].tolist(), 2)
    return img

def _timer() -> tuple:
    t = time()
    duration = 0
    def check():
        nonlocal t, duration
        t_tmp = time()
        duration = t_tmp - t
        t = t_tmp
    
    def elapsed():
        nonlocal duration
        return duration

    return check, elapsed

def _main():
    args: _Args = parse_args(_Args)
    args.class_inds = [int(x) for x in args.class_inds.split(",")]
    class_inds_tensor = torch.tensor(args.class_inds, device=args.device)

    if args.debug:
        seq_names = ["20171213_09/20171213_135628_001_Camera1"]
        timer_check, timer_elapsed = _timer()
        args.save_vis = True
        args.save_dets = True
    else:
        seq_names = load_seq_names_with_braking(args.path_to_db)
        end_idx = args.seq_end_idx if args.seq_end_idx != -1 else len(seq_names)
        seq_names = seq_names[args.seq_start_idx:end_idx]
        seq_names = ['20171213_01/20171215_201947_001_Camera1', '20171213_05/20171213_142144_001_Camera1']

    path_to_output = DATA_ROOT / "instance_segmentation"
    if args.save_dets or args.save_vis:
        path_to_output.mkdir(parents=True, exist_ok=True)

    detector = _Detector(args.detector_cfg_url, args.device)
    depth_estimator = _DepthEstimator(args.path_to_depth_ckpt, K[0, 0], 640, 320, device=args.device)

    db_cnx = sqlite3.connect(args.path_to_db)
    new_columns = ["dist_mean_1secs", "delta_dist_mean_1secs", "n_dets_1secs", "dist_mean_3secs", "delta_dist_mean_3secs", "n_dets_3secs"]
    new_column_default_vals = [-1, 0, 0, -1, 0, 0]
    for i, new_column in enumerate(new_columns):
        sql = f"ALTER TABLE probe_data ADD COLUMN {new_column} REAL default {new_column_default_vals[i]};"
        try:
            db_cnx.execute(sql)
        except Exception as e:
            print(e)
    try:
        db_cnx.execute(sql)
    except Exception as e:
        print(e)

    for seq_idx, seq_name in enumerate(seq_names):

        if args.debug:
            timer_check()

        csv_id, video_name = seq_name.split('/')

        sql = f"SELECT datetime FROM probe_data WHERE csv_identifier = '{csv_id}' AND seq_name = '{video_name}';"
        df = pd.read_sql(sql, db_cnx)
        df["datetime"] = pd.to_datetime(df["datetime"])
        frames, n_frames = dts_to_frames(DATA_ROOT, csv_id, video_name, df["datetime"], df["datetime"].min())
        frames = torch.tensor(np.round(frames).astype(int), device=args.device)

        db_cursor = db_cnx.cursor()

        dataset = KyushuDataset(csv_id, video_name, args.device)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
            num_workers=args.num_workers, collate_fn=_collate_fn, multiprocessing_context="spawn")

        dist_means = torch.ones(n_frames, device=args.device) * -1
        n_dets = torch.zeros(n_frames, device=args.device)
        if args.save_dets:
            dets = []
            path_to_output_seq = path_to_output / csv_id / video_name
            path_to_output_seq.mkdir(parents=True, exist_ok=True)
            path_to_output_seq_depth = DATA_ROOT / "depth_bts" / csv_id / video_name
            path_to_output_seq_depth.mkdir(parents=True, exist_ok=True)
        if args.save_vis:
            path_to_output_seq_vis = path_to_output / "vis" / csv_id / video_name
            path_to_output_seq_vis.mkdir(parents=True, exist_ok=True)

        if args.debug:
            timer_check()
            print(f"Preprocessing elapsed: {timer_elapsed():.4f}")

        pbar = tqdm(dataloader)
        pbar.set_description(f"{seq_idx+1}/{len(seq_names)}:{seq_name}")
        for batch_idx, batch_imgs in enumerate(tqdm(dataloader)):
            if args.debug:
                timer_check()
                print(f"Get data elapsed: {timer_elapsed():.4f}")
            with torch.no_grad():
                outputs = detector.forward(batch_imgs)
                if args.debug:
                    timer_check()
                    print(f"Detect elapsed: {timer_elapsed():.4f}")
                depths = depth_estimator.forward(torch.stack([x["image"] / 255.0 for x in batch_imgs]))
                if args.debug:
                    timer_check()
                    print(f"Depth estimation elapsed: {timer_elapsed():.4f}")
            
                img_inds = []
                classes = []
                scores = []
                masks = []
                if args.save_dets:
                    box2ds = []
                for i, output in enumerate(outputs):
                    n_dets_output = len(output["instances"])
                    if n_dets_output == 0:
                        continue
                    img_idx = batch_idx * args.batch_size + i
                    img_inds.append(torch.ones(n_dets_output, dtype=int, device=args.device) * img_idx)
                    classes.append(output["instances"].pred_classes)
                    scores.append(output["instances"].scores)
                    masks.append(output["instances"].pred_masks)
                    if args.save_dets:
                        box2ds.append(output["instances"].pred_boxes.tensor)
                if len(img_inds) > 0:
                    img_inds = torch.cat(img_inds) # (n_total_dets, )
                    classes = torch.cat(classes) # (n_total_dets, )
                    scores = torch.cat(scores) # (n_total_dets, )
                    masks = torch.cat(masks) # (n_total_dets, H, W)
                    if args.save_dets:
                        box2ds = torch.cat(box2ds) # (n_total_dets, 4)

                    keep = (scores >= args.score_threshold) & (torch.isin(classes, class_inds_tensor)) # (n_total_dets, )

                    img_inds = img_inds[keep] # (n_valid_dets, )
                    classes = classes[keep] # (n_valid_dets, )
                    scores = scores[keep] # (n_valid_dets, )
                    masks = masks[keep] # (n_valid_dets, H, W)
                    if args.save_dets:
                        box2ds = box2ds[keep] # (n_valid_dets, 4)

                for i in range(len(batch_imgs)):
                    img_idx = batch_idx * args.batch_size + i
                    if isinstance(img_inds, torch.Tensor) and (img_inds == img_idx).any():
                        mask = torch.any(masks[img_inds == img_idx], dim=0) # (H, W)
                        dist_means[img_idx] = torch.mean(depths[i, 0, mask])
                        n_dets[img_idx] = torch.sum(img_inds == img_idx)
                    if torch.isin(img_idx, frames).any():
                        dt = df["datetime"].iloc[torch.where(frames == img_idx)[0][0].item()]
                        dt_str = dt.strftime(DATETIME_FMT)
                        frames_before = [5, 15]
                        for n_frames_before in frames_before:
                            n_secs_before = n_frames_before // 5
                            start_img_idx = np.maximum(0, img_idx - n_frames_before)
                            dist_mean_n_secs = torch.mean(dist_means[start_img_idx:img_idx+1])
                            delta_dist_mean_n_secs = (dist_means[img_idx] - dist_means[start_img_idx]) / n_secs_before
                            n_dets_n_secs = torch.mean(n_dets[start_img_idx:img_idx+1])
                            sql = f"UPDATE probe_data SET dist_mean_{n_secs_before}secs = {dist_mean_n_secs}, delta_dist_mean_{n_secs_before}secs = {delta_dist_mean_n_secs}, n_dets_{n_secs_before}secs = {n_dets_n_secs} WHERE csv_identifier = '{csv_id}' AND seq_name = '{video_name}' AND datetime = '{dt_str}';"
                            db_cursor.execute(sql)

                if args.save_dets:
                    for i in range(args.batch_size):
                        img_idx = batch_idx * args.batch_size + i
                        if not (img_inds == img_idx).any():
                            dets.append({
                                "frame_id": img_idx,
                                "box2d": [],
                                "class": [],
                                "scores": [],
                            })
                            mask = np.zeros(masks.shape[1:], dtype=np.uint8)
                        else:
                            dets.append({
                                "frame_id": img_idx,
                                "box2ds": box2ds[img_inds == img_idx].detach().cpu().numpy(),
                                "classes": classes[img_inds == img_idx].detach().cpu().numpy(),
                                "scores": scores[img_inds == img_idx].detach().cpu().numpy(),
                            })
                            mask = torch.any(masks[img_inds == img_idx], dim=0).byte().detach().cpu().numpy()
                        Image.fromarray(mask).save(path_to_output_seq / f"{img_idx:06d}.png")
                
                if args.save_vis:
                    for i in range(len(batch_imgs)):
                        img_idx = batch_idx * args.batch_size + i
                        img_np = batch_imgs[i]["image"].permute(1, 2, 0).detach().cpu().numpy()
                        if (img_inds == img_idx).any():
                            det = dets[img_idx]
                            mask = torch.any(masks[img_inds == img_idx], dim=0).long().detach().cpu().numpy() # (H, W)
                            img_np = _draw_dets(img_np, det["box2ds"], mask, det["classes"], det["scores"], detector.thing_classes)
                        draw_text(img_np, f"{dist_means[img_idx]:.3f}")
                        Image.fromarray(img_np).save(path_to_output_seq_vis / f"{img_idx:06d}.png")
                        Image.fromarray((depths[i, 0] * 1000).detach().cpu().numpy().astype(np.uint16)).save(path_to_output_seq_depth / f"{img_idx:06d}.png")
            if args.debug:
                timer_check()
                print(f"Postprocessing elapsed: {timer_elapsed():.4f}")
        db_cnx.commit()
        if args.save_dets:
            save_json(dets, path_to_output / csv_id / f"{video_name}.json")

    db_cnx.close()

if __name__ == "__main__":
    _main()
