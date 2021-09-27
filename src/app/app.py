from flask import Flask, request, render_template, flash
from PIL import Image, ImageDraw, ImageFont
import sys
from gevent.pywsgi import WSGIServer
from flask_bootstrap import Bootstrap
import io, base64, os
import torch
from torchvision import transforms
from torchvision.ops import nms


app = Flask(__name__)
bootstrap = Bootstrap(app)

app.jinja_env.globals.update(zip=zip)

app.secret_key = 'aml projekt super secret key'

sys.path.append('../')
sys.path.append('../yolo/models')

from rcnn.model import ChestRCNN
from detection_fusion import EnsembleModel
from yolo.yolo import Model
from yolo.utils.general import non_max_suppression
from study.model import CovidModel

BACKBONE_PATH = os.environ['BACKBONE_PATH'] if 'BACKBONE_PATH' in os.environ else '../resnet/models/resnext101_32x8d_epoch_35.pt'
RCNN_STATE_DICT = os.environ['RCNN_STATE_DICT'] if 'RCNN_STATE_DICT' in os.environ else '../rcnn/models/fasterrcnn_epoch_23.pt'
YOLO_FINAL_MODEL_PATH = os.environ['YOLO_FINAL_MODEL_PATH'] if 'YOLO_FINAL_MODEL_PATH' in os.environ else "../yolo/models/yolov5_epoch_26.pt"
STUDY_STATE_DICT = os.environ['STUDY_STATE_DICT'] if 'STUDY_STATE_DICT' in os.environ else '../study/models/study_resnext_SGD_lr0-0005_m0-9_cos0-05_b15_epoch58.pt'

YOLO_CONFIG_PATH = os.environ['YOLO_CONFIG_PATH'] if 'YOLO_CONFIG_PATH' in os.environ else "../yolo/yolo5l.yaml"

INFERENCE_SIZE = 1024
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.2


rcnn_transforms = transforms.Compose([
                    transforms.Resize(INFERENCE_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


def load_torch_models():
    global fasterRCNN
    global yolo
    global ensemble
    global study_level

    fasterRCNN = ChestRCNN(BACKBONE_PATH).to('cpu')
    fasterRCNN.load_state_dict(torch.load(RCNN_STATE_DICT, map_location=torch.device('cpu')))
    fasterRCNN.eval()

    yolo = Model(cfg=YOLO_CONFIG_PATH,ch=3,nc=1)
    yolov5_weights = torch.load(YOLO_FINAL_MODEL_PATH, map_location=torch.device('cpu'))
    yolo.load_state_dict(yolov5_weights, strict=False)
    yolo.eval()

    ensemble = EnsembleModel(fasterRcnn=fasterRCNN, yolo=yolo, inference_size=INFERENCE_SIZE, iou_threshold_nms=IOU_THRESHOLD, confidence_threshold_nms=CONFIDENCE_THRESHOLD)

    study_level = CovidModel()
    study_level.load_state_dict(torch.load(STUDY_STATE_DICT, map_location=torch.device('cpu')))
    study_level.eval()

def inference_rcnn(img: Image):

    orig_width, orig_height = img.size
    width_factor = orig_width / INFERENCE_SIZE
    height_factor = orig_height / INFERENCE_SIZE

    #img_resized = img.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.ANTIALIAS)

    tensor_img = rcnn_transforms(img.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        out = fasterRCNN(tensor_img)
        out_indices = nms(boxes = out[0]['boxes'], scores=out[0]['scores'], iou_threshold=IOU_THRESHOLD)
        out_boxes = torch.index_select(out[0]['boxes'], 0, out_indices).detach().numpy()
        out_scores = torch.index_select(out[0]['scores'], 0, out_indices).detach().tolist()

    ret_boxes = []
    ret_scores = []

    if len(out_boxes) != 0:
        # Scale boxes back to original size
        ret_boxes = resize_boxes(out_boxes, width_factor, height_factor)
        ret_scores = [round(score, 4) for score in out_scores]

    return ret_boxes, ret_scores

def inference_yolo(img: Image):

    orig_width, orig_height = img.size
    width_factor = orig_width / INFERENCE_SIZE
    height_factor = orig_height / INFERENCE_SIZE

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

    if len(out_boxes) != 0:
        # Scale boxes back to original size
        ret_boxes = resize_boxes(out_boxes, width_factor, height_factor)
        ret_scores = [round(score, 4) for score in out_scores]

    return ret_boxes, ret_scores


def inference_ensemble(img:Image):
    orig_width, orig_height = img.size
    width_factor = orig_width / INFERENCE_SIZE
    height_factor = orig_height / INFERENCE_SIZE

    img_resized = img.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.ANTIALIAS)

    _, out_boxes, out_scores = ensemble.detection_fusion(rcnn_transforms(img.convert("RGB")).unsqueeze(0), transforms.ToTensor()(img_resized).unsqueeze_(0), extended_output=False)

    ret_boxes = []
    ret_scores = []

    if len(out_boxes) != 0:
        # Scale boxes back to original size
        ret_boxes = resize_boxes(out_boxes, width_factor, height_factor)
        ret_scores = [round(score, 4) for score in out_scores]

    return ret_boxes, ret_scores

def inference_study(img:Image):

    tensor_img = rcnn_transforms(img.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        out = study_level(tensor_img)

    return out.detach().tolist()

def resize_boxes(boxes, width_factor, height_factor):
    return [[round(box[0] * width_factor, 4), round(box[1] * height_factor, 4), round(box[2] * width_factor, 4), round(box[3] * height_factor, 4)] for box in boxes]

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
def inference_form():

    if 'file' not in request.files.keys() or request.files['file'].mimetype == 'application/octet-stream':
        alert = "Please provide a valid image. Also note that an image that was already submitted cannot be re-submitted and has to be uploaded again."
        flash(alert)
        return render_template('index.html')

    img = Image.open(request.files['file'])

    model_select = list(request.form.keys())[0]

    img = img.convert('RGB')

    if model_select == 'rcnn': 
        ret_boxes, ret_scores = inference_rcnn(img)
    elif model_select == 'yolo':
        ret_boxes, ret_scores = inference_yolo(img)
    elif model_select == 'ensemble':
        ret_boxes, ret_scores = inference_ensemble(img)

    study_out = inference_study(img)

    draw = ImageDraw.Draw(img)
    
    for i, box in enumerate(ret_boxes):
        draw.rectangle(box, outline="#FF0000", width=5)
        draw.text((box[0]+20, box[1]), str(i+1), fill="#FF0000", font=ImageFont.truetype("./Arial.ttf", 100))

    img_io = io.BytesIO()

    img.save(img_io, 'PNG', quality=100)
    img_io.seek(0)
    img = base64.b64encode(img_io.getvalue())

    return render_template('index.html', boxes=ret_boxes, scores=ret_scores, img=img.decode('ascii'), study_out=study_out)

print(("* Loading PyTorch models and starting Flask server..."))
load_torch_models()
print("* Models loaded")
# Run app
app_server = WSGIServer(("0.0.0.0", 5000), app)
app_server.serve_forever()