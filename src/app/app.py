from flask import Flask, request, render_template
from flask_httpauth import HTTPBasicAuth
from PIL import Image, ImageDraw
import sys
from gevent.pywsgi import WSGIServer
from flask_bootstrap import Bootstrap
import io, base64
import torch
from torchvision import transforms
from torchvision.ops import nms


app = Flask(__name__)
bootstrap = Bootstrap(app)
auth = HTTPBasicAuth()

app.jinja_env.globals.update(zip=zip)

sys.path.append('../')

from rcnn.model import ChestRCNN
from detection_fusion import EnsembleModel
from yolo.yolo import Model
from yolo.utils.general import non_max_suppression

BACKBONE_PATH = '../resnet/models/resnext101_32x8d_epoch_35.pt'
RCNN_STATE_DICT = '../rcnn/models/fasterrcnn_epoch_23.pt'
YOLO_FINAL_MODEL_PATH = "./yolo/models_final_giou_40/yolov5_epoch_25.pt"


INFERENCE_SIZE = 1024
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.2


def load_torch_models():
    global fasterRCNN
    global yolo

    fasterRCNN = ChestRCNN(BACKBONE_PATH).to('cpu')
    fasterRCNN.load_state_dict(torch.load(RCNN_STATE_DICT, map_location=torch.device('cpu')))
    fasterRCNN.eval()

    yolo = Model(cfg="./yolo/yolo5l.yaml",ch=3,nc=1)
    yolov5_weights = torch.load(YOLO_FINAL_MODEL_PATH)
    yolo.load_state_dict(yolov5_weights, strict=False)
    yolo.eval()

def inference_rcnn(img: Image):

    orig_width, orig_height = img.size
    width_factor = orig_width / INFERENCE_SIZE
    height_factor = orig_height / INFERENCE_SIZE

    img_resized = img.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.ANTIALIAS)

    tensor_img = transforms.ToTensor()(img_resized).unsqueeze_(0)

    with torch.inference_mode():
        out = fasterRCNN(tensor_img)
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
        ret_boxes = [[round(box[0] * width_factor, 4), round(box[1] * height_factor, 4), round(box[2] * width_factor, 4), round(box[3] * height_factor, 4)] for box in out_boxes]
        ret_scores = [round(score, 4) for score in out_scores]

    return ret_boxes, ret_scores


def inference_ensemble(img:Image):
    _, boxes, scores = EnsembleModel(fasterRCNN=fasterRCNN, yolo=yolo, inference_size=INFERENCE_SIZE, iou_threshold_nms=IOU_THRESHOLD, confidence_threshold_nms=CONFIDENCE_THRESHOLD).detection_fusion(img, extended_output=False)
    return boxes, scores


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/", methods=['POST'])
@auth.login_required
def inference_form():

    if 'file' not in request.files.keys():
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

    draw = ImageDraw.Draw(img)
    
    for box in ret_boxes:
        draw.rectangle(box, outline="#FF0000", width=5)

    img_io = io.BytesIO()

    img.save(img_io, 'PNG', quality=100)
    img_io.seek(0)
    img = base64.b64encode(img_io.getvalue())

    return render_template('index.html', boxes=ret_boxes, scores=ret_scores, img=img.decode('ascii'))

@app.route("/inference", methods=['POST'])
@auth.login_required
def inference_api():

    if 'file' in request.files.keys():
        img = request.files['file']
    else:
        return {'error': 'Request not in correct format. Send an image with the name "file"'}, 400

    ret_boxes, ret_scores = inference_rcnn(Image.open(img))

    return {'boxes': ret_boxes, 'scores': ret_scores}, 200

# For now just use hardcoded credentials
@auth.get_password
def get_password(InputUsername):
    if InputUsername == "amlproject":
        return "amlproject"
    return None

print(("* Loading PyTorch models and starting Flask server..."))
load_torch_models()
print("* Models loaded")
# Run app
app_server = WSGIServer(("0.0.0.0", 5000), app)
app_server.serve_forever()