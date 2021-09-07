from numba.cuda.args import In
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


YOLO_FINAL_MODEL_PATH = "./yolo/models_final_giou_10/yolov5_epoch_14.pt"
FASTER_RCNN_FINAL_MODEL_PATH = "./rcnn/models/fasterrcnn_epoch_23.pt"
RESNET_BACKBONE_PATH = "./resnet/models/resnext101_32x8d_epoch_35.pt"

IOU_THRESHOLD_FUSION = .2
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.2
INFERENCE_SIZE = 512

def scale_boxes_to_size(width_factor,height_factor, out_boxes):
    return [[round(box[0] * width_factor, 4), round(box[1] * height_factor, 4), round(box[2] * width_factor, 4), round(box[3] * height_factor, 4)] for box in out_boxes]

def normalize_boxes(box_pred):
    return [[round(box[0] /INFERENCE_SIZE, 4), round(box[1] / INFERENCE_SIZE, 4), round(box[2] /INFERENCE_SIZE, 4), round(box[3] / INFERENCE_SIZE, 4)] for box in box_pred]

def refine_det(boxes):
    boxes_out = []
    for box in boxes:
        x1, y1, x2, y2 = box
        if x1==x2 or y1==y2:
            continue
        box = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
        boxes_out.append(box)

    return boxes_out

def inference_rcnn(img: Image, fasterRcnn:ChestRCNN):

    img_resized = img.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.ANTIALIAS)

    tensor_img = transforms.ToTensor()(img_resized).unsqueeze_(0)
    with torch.inference_mode():
        out = fasterRcnn(tensor_img)
        out_indices = nms(boxes = out[0]['boxes'], scores=out[0]['scores'], iou_threshold=IOU_THRESHOLD)
        out_boxes = torch.index_select(out[0]['boxes'], 0, out_indices).detach().numpy()
        out_scores = torch.index_select(out[0]['scores'], 0, out_indices).detach().tolist()

    ret_boxes = []
    ret_scores = []
    ret_labels = []

    if len(out_boxes) != 0:
        # Scale boxes back to [0,1]
        ret_boxes = normalize_boxes(out_boxes)
        ret_scores = [round(score, 4) for score in out_scores]
        ret_labels = [1 for score in ret_scores] # label is always 1


    return refine_det(ret_boxes), ret_scores, ret_labels


def inference_yolo(img: Image, yolo:Model):
    # Tbd: img /=255 
    img_resized = img.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.ANTIALIAS)

    tensor_img = transforms.ToTensor()(img_resized).unsqueeze_(0)

    with torch.inference_mode():
        prediction = yolo(tensor_img,augment=True)[0]

    prediction = non_max_suppression(prediction, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, classes=None)[0]
    out_boxes = prediction[:,:4].detach().numpy()
    out_scores = prediction[:,4].detach().tolist()

    ret_boxes = []
    ret_scores = []
    ret_labels = []


    if len(out_boxes) != 0:
        # Scale boxes back to [0,1]
        ret_boxes = normalize_boxes(out_boxes)
        ret_scores = [round(score, 4) for score in out_scores]
        ret_labels = [1 for score in ret_scores] # label is always 1

    return refine_det(ret_boxes), ret_scores, ret_labels

def detection_fusion(img, extended_output=False):

    img = img.convert('RGB')

    #gather image information
    orig_width, orig_height = img.size

    # Load both models
    yolov5_weights = torch.load(YOLO_FINAL_MODEL_PATH)
    fasterrcnn_r101_weights = torch.load(FASTER_RCNN_FINAL_MODEL_PATH)

    yolo = Model(cfg="./yolo/yolo5l.yaml",ch=3,nc=1)
    yolo.load_state_dict(yolov5_weights, strict=False) 

    fasterRcnn = ChestRCNN(RESNET_BACKBONE_PATH)
    fasterRcnn.load_state_dict(fasterrcnn_r101_weights)

    yolo.eval()
    fasterRcnn.eval()

    # inference 
    frcnn_boxes, frcnn_scores, frcnn_labels = inference_rcnn(img, fasterRcnn)
    yolo_boxes, yolo_scores, yolo_labels = inference_yolo(img, yolo)

    boxes = [frcnn_boxes, yolo_boxes]
    scores = [frcnn_scores , yolo_scores]
    
    labels = [frcnn_labels, yolo_labels]
    
    weights = [1,1]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=IOU_THRESHOLD_FUSION)
    boxes = boxes.clip(0,1)

    # scale box back to original size
    boxes = scale_boxes_to_size(orig_width, orig_height, boxes)

    opacity_pred = []
    for box, score in zip(boxes, scores):
        opacity_pred.append('opacity {} {} {} {} {}'.format(score, box[0], box[1], box[2],box[3]))

    if extended_output:
        return opacity_pred, boxes, scores, labels, scale_boxes_to_size(yolo_boxes), yolo_scores, scale_boxes_to_size(frcnn_boxes), frcnn_scores

    return opacity_pred, boxes, scores


if __name__ == "__main__":
    data_path = '../data/siim-covid19-detection/data/test/00a81e8f1051/bdc0bb04ae1e/ced40f593496.png'
    img = Image.open(data_path)
    detection_fusion(img)