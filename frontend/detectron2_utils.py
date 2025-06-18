#detectron2_utils.py
import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger

setup_logger()

def setup_detectron2(
    model_path: str = "model_final.pth",
    num_classes:    int  = 2,
    score_thresh:   float = 0.5
) -> DefaultPredictor:

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS                     = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES       = num_classes
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup_detectron2] device → {device}")
    cfg.MODEL.DEVICE = device

    return DefaultPredictor(cfg)


def process_frame(frame: np.ndarray, predictor: DefaultPredictor) -> np.ndarray:

    inst    = predictor(frame)["instances"].to("cpu")
    boxes   = inst.pred_boxes.tensor.numpy()
    classes = inst.pred_classes.numpy()
    scores  = inst.scores.numpy()

    thresh = predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST

    for (x1, y1, x2, y2), cls, score in zip(boxes, classes, scores):
        if score < thresh:
            continue

        # choose color by class
        col   = (0, 170, 170) if cls == 0 else (170, 0, 0)
        label = f"{cls}:{score:.2f}"

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            col, 2
        )
        cv2.putText(
            frame,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            col,
            2
        )

    return frame

