## Covid-19 detection Kaeggle challenge source files

### Folders

- [/app](./app): Code for the webapp
- [/rcnn](./rcnn): Includes code to train and evaluate the Faster R-CNN.
- [/resnet](./resnet): ResNeXt backbone training code.
- [/study](./study): Study-level model.
- [/yolo](./yolo): Yolo object detection model code

The training for all models can be started with: `python train.py`. They all include checkpointing ability and can be resumed at any time

### Files

- [data-preparation.ipynb](./data-preparation.ipynb): Dataset exploration and conversion from DICOM to more usable formats
- [data-statistics.ipynb](./data-statistics.ipynb): Examples of class distributions of the datasets
- [model-evaluations.ipynb](./model-evaluations.ipynb): Loss statistics and evaluation metrics of all models except the ensemble
- [fusion_test.ipynb](./fusion_test.ipynb): Ensemble model evaluation
- [Dockerfile](./Dockerfile): Dockerfile to build the webapp. Build with `docker build -t <your tag here> .` and run with `docker run --rm -p 5000:5000 <your tag here>`. Pre-built images use the tag `tobiasrst/aml-project`