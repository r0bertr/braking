import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from matplotlib import pyplot as plt
from torch import Tensor


class ObjectDetector:
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

    def draw_dets(self, img: np.ndarray, box2d: np.ndarray, mask: np.ndarray,
        classes: np.ndarray, scores: np.ndarray) -> None:

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
            class_name = self.thing_classes[class_idx]
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[i+1].tolist(), 2)
            cv2.putText(img, f"{i+1}:{class_name}:{scores[i]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i+1].tolist(), 2)
        return img
