from flask import Flask, make_response, request, jsonify, abort
from flask_httpauth import HTTPBasicAuth
from PIL import Image
import sys
from gevent.pywsgi import WSGIServer

import torch
from torchvision import transforms
from torchvision.ops import nms

app = Flask(__name__)
auth = HTTPBasicAuth()

sys.path.append('../')
from rcnn.model import ChestRCNN

BACKBONE_PATH = '../resnet/models/resnext101_32x8d_epoch_35.pt'
RCNN_STATE_DICT = '../rcnn/models/fasterrcnn_epoch_23.pt'

INFERENCE_SIZE = 1024


def load_torch_model():
    global model
    model = ChestRCNN(BACKBONE_PATH).to('cpu')
    model.load_state_dict(torch.load(RCNN_STATE_DICT, map_location=torch.device('cpu')))
    model.eval()

@app.route("/inference", methods=['POST'])
@auth.login_required
def inference():
    if 'file' in request.files.keys():
        img = Image.open(request.files['file'])
    else:
        return {'error': 'Request not in correct format. Send an image with the name "file"'}, 400

    orig_width, orig_height = img.size
    width_factor = orig_width / INFERENCE_SIZE
    height_factor = orig_height / INFERENCE_SIZE

    img_resized = img.resize((INFERENCE_SIZE, INFERENCE_SIZE), Image.ANTIALIAS)

    tensor_img = transforms.ToTensor()(img_resized).unsqueeze_(0)

    with torch.inference_mode():
        out = model(tensor_img)
        out_indices = nms(boxes = out[0]['boxes'], scores=out[0]['scores'], iou_threshold=0.2)
        out_boxes = torch.index_select(out[0]['boxes'], 0, out_indices).detach().numpy()
        out_scores = torch.index_select(out[0]['scores'], 0, out_indices).detach().tolist()

    ret_boxes = []
    ret_scores = []

    if len(out_boxes) != 0:
        # Scale boxes back to original size
        ret_boxes = [[round(box[0] * width_factor, 4), round(box[1] * height_factor, 4), round(box[2] * width_factor, 4), round(box[3] * height_factor, 4)] for box in out_boxes]
        ret_scores = [round(score, 4) for score in out_scores]

    return {'boxes': ret_boxes, 'scores': ret_scores}, 200

# For now just use hardcoded credentials
@auth.get_password
def get_password(InputUsername):
    if InputUsername == "amlproject":
        return "amlproject"
    return None

print(("* Loading PyTorch model and starting Flask server..."))
load_torch_model()
print("* Model loaded")
# Run app
app_server = WSGIServer(("0.0.0.0", 5000), app)
app_server.serve_forever()