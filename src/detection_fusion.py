import numpy as np
import torch
import torch
from ensemble_boxes import weighted_boxes_fusion
from rcnn.model import ChestRCNN
from yolo.yolo import Model
import torch
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image
from yolo.utils.general import non_max_suppression


YOLO_FINAL_MODEL_PATH = "./yolo/models_final_giou_10/yolov5_epoch_10.pt"
FASTER_RCNN_FINAL_MODEL_PATH = "./rcnn/models/fasterrcnn_epoch_50.pt"
RESNET_BACKBONE_PATH = "../resnet/models/resnext101_32x8d_epoch_35.pt"

IOU_THRESHOLD_FUSION = .6
CONFIDENCE_THRESHOLD = 0.01
IOU_THRESHOLD = 0.2
INFERENCE_SIZE = 1024



def inference_rcnn(img: Image, fasterRcnn:ChestRCNN):

    orig_width, orig_height = img.size
    width_factor = orig_width / INFERENCE_SIZE
    height_factor = orig_height / INFERENCE_SIZE

    img_resized = img.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.ANTIALIAS)

    tensor_img = transforms.ToTensor()(img_resized).unsqueeze_(0)

    with torch.inference_mode():
        out = fasterRcnn(tensor_img)
        out_indices = nms(boxes = out[0]['boxes'], scores=out[0]['scores'], iou_threshold=IOU_THRESHOLD)
        out_boxes = torch.index_select(out[0]['boxes'], 0, out_indices).detach().numpy()
        out_scores = torch.index_select(out[0]['scores'], 0, out_indices).detach().tolist()

    ret_boxes = []
    ret_scores = []

    if len(out_boxes) != 0:
        # Scale boxes back to original size
        ret_boxes = [[round(box[0] * width_factor, 4), round(box[1] * height_factor, 4), round(box[2] * width_factor, 4), round(box[3] * height_factor, 4)] for box in out_boxes]
        ret_scores = [round(score, 4) for score in out_scores]

    return ret_boxes, ret_scores


def inference_yolo(img: Image, yolo:Model):
    orig_width, orig_height = img.size
    width_factor = orig_width / INFERENCE_SIZE
    height_factor = orig_height / INFERENCE_SIZE

    img_resized = img.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.ANTIALIAS)

    tensor_img = transforms.ToTensor()(img_resized).unsqueeze_(0)

    with torch.inference_mode():
        prediction = yolo(tensor_img,augment=True)[0]

    prediction = non_max_suppression(prediction, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, classes=None)[0]
    out_boxes = prediction[:,:4]
    out_scores = prediction[:,4]
    ret_boxes = []
    ret_scores = []

    if len(out_boxes) != 0:
        # Scale boxes back to original size
        ret_boxes = [[round(box[0] * width_factor, 4), round(box[1] * height_factor, 4), round(box[2] * width_factor, 4), round(box[3] * height_factor, 4)] for box in out_boxes]
        ret_scores = [round(score, 4) for score in out_scores]

    return ret_boxes, ret_scores

def detecion_fusion(img):

    # Load both models
    yolov5_weights = torch.load(YOLO_FINAL_MODEL_PATH)
    fasterrcnn_r101_weights = torch.load(FASTER_RCNN_FINAL_MODEL_PATH)

    yolo = Model(cfg="yolo5l.yaml",ch=3,nc=1)
    yolo.load_state_dict(yolov5_weights, strict=False) 

    fasterRcnn = ChestRCNN(RESNET_BACKBONE_PATH)
    fasterRcnn.load_state_dict(fasterrcnn_r101_weights)

    # inference 
    frcnn_boxes, frcnn_scores = inference_rcnn(img, fasterRcnn)
    yolo_boxes, yolo_scores = inference_yolo(img, yolo)

    boxes = frcnn_boxes + yolo_boxes
    labels = ['opacity', 'opacity']
    scores = frcnn_scores + yolo_scores

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=IOU_THRESHOLD_FUSION)
    boxes = boxes.clip(0,1)

    opacity_pred = []
    for box, score in zip(boxes, scores):
        opacity_pred.append('opacity {} {} {} {} {}'.format(score, box[0], box[1], box[2],box[3]))

    return opacity_pred