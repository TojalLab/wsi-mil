import sys
import torch
import torch.utils.data
import os.path
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcallbacks
from omegaconf import open_dict

from aim.pytorch_lightning import AimLogger

import ray





def main(cfg):
    print(cfg)
    print()

    dataset_meta = (os.path.splitext(os.path.basename(cfg.common.dataset_metadata))[0]).replace("-", "_")
    basepath = os.path.basename(os.getcwd()).replace("-","_")
    fold_idx = cfg.train_model.dataset.fold
    exp_name = f'{basepath}-{dataset_meta}-{cfg.train_model.model_type}-{cfg.common.target_label}-fold{fold_idx}'
    print(exp_name)
