FROM jjanzic/docker-python3-opencv

ENV BACKBONE_PATH="/app/resnet/models/resnext101_32x8d_epoch_35.pt"
ENV RCNN_STATE_DICT="/app/rcnn/models/fasterrcnn_epoch_23.pt"
ENV YOLO_FINAL_MODEL_PATH="/app/yolo/models/yolov5_epoch_26.pt"
ENV STUDY_STATE_DICT="/app/study/models/study_resnext_SGD_lr0-0005_m0-9_cos0-05_b15_epoch58.pt"


WORKDIR /app/web

COPY app/requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY app/ /app/web
COPY resnet/ /app/resnet
COPY rcnn/ /app/rcnn
COPY yolo/ /app/yolo
COPY study/ /app/study

COPY detection_fusion.py /app/

# RUN mkdir -p /app/models

# COPY resnet/models/resnext101_32x8d_epoch_35.pt ${BACKBONE_PATH}
# COPY rcnn/models/fasterrcnn_epoch_23.pt ${RCNN_STATE_DICT}
# COPY yolo/models/yolov5_epoch_26.pt ${YOLO_FINAL_MODEL_PATH}
# COPY study/models/study_resnext_SGD_lr0-0005_m0-9_cos0-05_b15_epoch58.pt ${STUDY_STATE_DICT}

EXPOSE 5000

CMD [ "python3", "-u", "app.py"]


