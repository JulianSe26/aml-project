import numpy as np
import pandas as pd
import torch
import os
import torch
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion



YOLO_FINAL_MODEL_PATH = "../detection_yolov5/predictions/yolov5x6_768_fold0_1_2_3_4_{}_pred.pth"
FASTER_RCNN_FINAL_MODEL_PATH = "../detection_fasterrcnn/predictions/resnet101d_1024_fold0_1_2_3_4_{}_pred.pth"

IOU_THRESHOLD_FUSION = .6


if __name__ == "__main__":
    os.makedirs('../../dataset/pseudo_csv_det', exist_ok=True)

    # Load both models
    yolov5_768_image_pred = torch.load(YOLO_FINAL_MODEL_PATH)
    fasterrcnn_r101_image_pred = torch.load(FASTER_RCNN_FINAL_MODEL_PATH)

    boxes1, scores1, labels1, img_width1, img_height1 = yolov5_768_image_pred[row['imageid']]
    boxes2, scores2, labels2, img_width2, img_height2 = fasterrcnn_r101_image_pred[row['imageid']]

    boxes = boxes1 + boxes2
    labels = labels1 + labels2
    scores = scores1 + scores2

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=IOU_THRESHOLD_FUSION)
    boxes = boxes.clip(0,1)