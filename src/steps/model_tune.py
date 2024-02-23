import sys
import torch
import os
import torch.utils.data
import os.path
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcallbacks
from omegaconf import open_dict

from aim.pytorch_lightning import AimLogger
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.search.ax import AxSearch

from lib.etc import read_metadata
from lib.models.clam_interface import CLAMInterface
from lib.models.dataset_interface import DataModule
from lib.models.transmil_interface import TransMILInterface
from lib.models.dataset2_interface import TileDataModule
from lib.models.dataset3_inferface import AugFeaturesModule
from lib.models.cnn_interface import CNNInterface
from lib.models.mamil_interface import MAMILInterface

from .train_model import train_one_fold

def main(cfg):
    config = {
        #'weighted_sample': tune.choice(['oversample','undersample',None]),
        #'weighted_loss': tune.choice([True, False]),
        'lr': tune.choice([1e-5,3e-5,1e-4,3e-4,1e-3,3e-3]),
        'dropout': tune.choice([0.0, 0.2, 0.4, 0.6, 0.8]),
        'wd': tune.choice([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
        'accumulate_grad_batches': tune.choice([1, 5, 10, 15, 20]),
        'optim': tune.choice(['adam','adamw'])
    }

    def trainable(extra):
        with open_dict(cfg):
            cfg.train_model.trainer.tune_callback = True
            cfg.train_model.dataset.weighted_sample = 'oversample' # extra['weighted_sample']
            cfg.train_model.model.weighted_loss = False # extra['weighted_loss']
            cfg.train_model.CLAM.learning_rate = extra['lr']
            cfg.train_model.CLAM.weight_decay = extra['wd']
            cfg.train_model.dataset.dropout = extra['dropout']
            cfg.train_model.CLAM.optim = extra['optim']
            cfg.train_model.trainer.accumulate_grad_batches = extra['accumulate_grad_batches']
        os.chdir(os.environ["TUNE_ORIG_WORKING_DIR"])
        train_one_fold(cfg)

    scheduler = ASHAScheduler(
        max_t=100,
        grace_period=20
    )

    analysis = tune.run(
        trainable,
        metric='tune_metric',
        mode='max',
        config=config,
        num_samples=50,
        search_alg=AxSearch(),
        scheduler=scheduler,
        resources_per_trial={ 'gpu': 0.5 }
    )
    print("Best hyperparameters found were: ", analysis.best_config)
