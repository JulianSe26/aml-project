#!/usr/bin/env bash

python ./param_train.py --checkpoint /home/tkrieger/var/aml-models/Resnet/resnext101_32x8d_epoch_35.pt --initial_ckpt True --config_module parameters.SGD_lr0-0001_m0-9_cos0-05_b15
python ./param_train.py --checkpoint /home/tkrieger/var/aml-models/Resnet/resnext101_32x8d_epoch_35.pt --initial_ckpt True --config_module parameters.SGD_lr0-0002_m0-9_cos0-05_b15
python ./param_train.py --checkpoint /home/tkrieger/var/aml-models/Resnet/resnext101_32x8d_epoch_35.pt --initial_ckpt True --config_module parameters.ADAM_lr0-001_cos0-05_b15
python ./param_train.py --checkpoint /home/tkrieger/var/aml-models/Resnet/resnext101_32x8d_epoch_35.pt --initial_ckpt True --config_module parameters.SGD_lr0-0001_m0-9_ScLf0-05_b15
python ./param_train.py --checkpoint /home/tkrieger/var/aml-models/Resnet/resnext101_32x8d_epoch_35.pt --initial_ckpt True --config_module parameters.SGD_lr0-0005_m0-9_cos0-05_b15
python ./param_train.py --checkpoint /home/tkrieger/var/aml-models/Resnet/resnext101_32x8d_epoch_35.pt --initial_ckpt True --config_module parameters.SGD_lr0-0001_m0-9_cos0-05_b35
