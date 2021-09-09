import sys
sys.path.append('./yolo')
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


class EnsembleModel():

    def __init__(self, fasterRcnn:ChestRCNN, yolo:Model, inference_size = 512, iou_threshold_nms=.2, iou_threshold_fusion=.55, confidence_threshold_nms=.1):
        self.fasterRcnn = fasterRcnn
        self.yolo = yolo
        self.inference_size = inference_size
        self.iou_threshold_nms = iou_threshold_nms
        self.iou_threshold_fusion = iou_threshold_fusion
        self.confidence_threhsold_nms = confidence_threshold_nms

    def scale_boxes_to_size(self,width_factor,height_factor, out_boxes):
        return [[round(box[0] * width_factor, 4), round(box[1] * height_factor, 4), round(box[2] * width_factor, 4), round(box[3] * height_factor, 4)] for box in out_boxes]

    def normalize_boxes(self,box_pred):
        return [[round(box[0] /self.inference_size, 4), round(box[1] / self.inference_size, 4), round(box[2] / self.inference_size, 4), round(box[3] / self.inference_size, 4)] for box in box_pred]

    def refine_det(self,boxes):
        boxes_out = []
        for box in boxes:
            x1, y1, x2, y2 = box
            if x1==x2 or y1==y2:
                continue
            box = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
            boxes_out.append(box)

        return boxes_out

    def inference_rcnn(self,img: Image):

        img_resized = img.resize((self.inference_size, self.inference_size), Image.ANTIALIAS)

        tensor_img = transforms.ToTensor()(img_resized).unsqueeze_(0)
        with torch.inference_mode():
            out = self.fasterRcnn(tensor_img)
            out_indices = nms(boxes = out[0]['boxes'], scores=out[0]['scores'], iou_threshold=self.iou_threshold_nms)
            out_boxes = torch.index_select(out[0]['boxes'], 0, out_indices).detach().numpy()
            out_scores = torch.index_select(out[0]['scores'], 0, out_indices).detach().tolist()

        ret_boxes = []
        ret_scores = []
        ret_labels = []

        if len(out_boxes) != 0:
            # Scale boxes back to [0,1]
            ret_boxes = self.normalize_boxes(out_boxes)
            ret_scores = [round(score, 4) for score in out_scores]
            ret_labels = [1 for score in ret_scores] # label is always 1


        return self.refine_det(ret_boxes), ret_scores, ret_labels


    def inference_yolo(self,img: Image):
        img_resized = img.resize((self.inference_size, self.inference_size), Image.ANTIALIAS)

        tensor_img = transforms.ToTensor()(img_resized).unsqueeze_(0)
        #tensor_img /= 255

        with torch.inference_mode():
            prediction = self.yolo(tensor_img,augment=True)[0]

        prediction = non_max_suppression(prediction, self.confidence_threhsold_nms, self.iou_threshold_nms, classes=None)[0]
        out_boxes = prediction[:,:4].detach().numpy()
        out_scores = prediction[:,4].detach().tolist()

        ret_boxes = []
        ret_scores = []
        ret_labels = []


        if len(out_boxes) != 0:
            # Scale boxes back to [0,1]
            ret_boxes = self.normalize_boxes(out_boxes)
            ret_scores = [round(score, 4) for score in out_scores]
            ret_labels = [1 for score in ret_scores] # label is always 1

        return self.refine_det(ret_boxes), ret_scores, ret_labels

    def detection_fusion(self,img, extended_output=False):

        img = img.convert('RGB')

        #gather image information
        orig_width, orig_height = img.size

        self.yolo.eval()
        self.fasterRcnn.eval()

        # inference 
        frcnn_boxes, frcnn_scores, frcnn_labels = self.inference_rcnn(img)
        yolo_boxes, yolo_scores, yolo_labels = self.inference_yolo(img)
        #print(f'YOLO results: {yolo_scores}, {yolo_labels}')

        boxes = [frcnn_boxes, yolo_boxes]
        scores = [frcnn_scores , yolo_scores]
        
        labels = [frcnn_labels, yolo_labels]
        
        weights = [1,1]
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=self.iou_threshold_fusion)
        boxes = boxes.clip(0,1)

        #TODO: maybe perform NMS again ?

        # scale box back to original size
        boxes = self.scale_boxes_to_size(orig_width, orig_height, boxes)

        opacity_pred = []
        for box, score in zip(boxes, scores):
            opacity_pred.append('opacity {} {} {} {} {}'.format(score, box[0], box[1], box[2],box[3]))

        if extended_output:
            return opacity_pred, boxes, scores, labels, self.scale_boxes_to_size(orig_width, orig_height, yolo_boxes), yolo_scores, yolo_labels, self.scale_boxes_to_size(orig_width, orig_height, frcnn_boxes), frcnn_scores, frcnn_labels

        return opacity_pred, boxes, scores


if __name__ == "__main__":

    YOLO_FINAL_MODEL_PATH = "./yolo/models_final_giou_40/yolov5_epoch_25.pt"
    FASTER_RCNN_FINAL_MODEL_PATH = "./rcnn/models/fasterrcnn_epoch_23.pt"
    RESNET_BACKBONE_PATH = "./resnet/models/resnext101_32x8d_epoch_35.pt"
    SAMPLE_DATA = '../data/siim-covid19-detection/data/test/00a81e8f1051/bdc0bb04ae1e/ced40f593496.png'

    img = Image.open(SAMPLE_DATA)
    yolov5_weights = torch.load(YOLO_FINAL_MODEL_PATH)
    fasterrcnn_r101_weights = torch.load(FASTER_RCNN_FINAL_MODEL_PATH)

    yolo = Model(cfg="./yolo/yolo5l.yaml",ch=3,nc=1)
    yolo.load_state_dict(yolov5_weights, strict=False) 

    fasterRcnn = ChestRCNN(RESNET_BACKBONE_PATH)
    fasterRcnn.load_state_dict(fasterrcnn_r101_weights)
    print(EnsembleModel(fasterRcnn=fasterRcnn, yolo=yolo).detection_fusion(img))