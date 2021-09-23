FROM jjanzic/docker-python3-opencv

ENV BACKBONE_PATH="/app/models/resnext101_32x8d_epoch_35.pt"
ENV RCNN_STATE_DICT="/app/models/fasterrcnn_epoch_23.pt"
ENV YOLO_FINAL_MODEL_PATH="/app/models/yolov5_epoch_26.pt"


WORKDIR /app/web

COPY app/requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY app/ /app/web
COPY resnet/ /app/resnet
COPY rcnn/ /app/rcnn
COPY yolo/ /app/yolo

COPY detection_fusion.py /app/

RUN mkdir -p /app/models

COPY resnet/models/resnext101_32x8d_epoch_35.pt ${BACKBONE_PATH}
COPY rcnn/models/fasterrcnn_epoch_23.pt ${RCNN_STATE_DICT}
COPY yolo/models/yolov5_epoch_26.pt ${YOLO_FINAL_MODEL_PATH}

EXPOSE 5000

CMD [ "python3", "app.py"]


