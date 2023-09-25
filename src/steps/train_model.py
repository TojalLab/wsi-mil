import sys
import torch
import torch.utils.data
import os.path
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plcallbacks
from omegaconf import open_dict

from aim.pytorch_lightning import AimLogger

import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from lib.etc import read_metadata
from lib.models.clam_interface import CLAMInterface
from lib.models.clam2_interface import CLAM2Interface
from lib.models.dataset_interface import DataModule
from lib.models.dataset2_interface import TileDataModule
from lib.models.dataset_interface import SlideBatchDataset
from lib.models.cnn_interface import CNNInterface
from lib.models.addmil_interface import ADDMILInterface

class ExitCallback(plcallbacks.Callback):
    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            print(f'upgrading KeyboardInterrupt to SystemExit')
            raise SystemExit()

def train_one_fold(cfg):
    tbl = read_metadata(cfg.train_model.input.metadata)
    if cfg.train_model.dataset.fold == 'all':
        raise Exception("cant run for all folds")

    fold_idx = cfg.train_model.dataset.fold
    metadata = tbl[tbl.fold==fold_idx]
    target_col = cfg.common.target_label
    metadata = metadata[~metadata[target_col].isnull()].reset_index(drop=True)

    if 'CLAM' in cfg.train_model.model_type:
        if cfg.train_model.dataset.augmented_sample:
            dm = TileDataModule(cfg, metadata)
            m = CLAM2Interface(cfg.train_model)
        else:
            dm = DataModule(cfg, metadata)
            m = CLAMInterface(cfg.train_model)
    elif cfg.train_model.model_type == 'CNN':
        dm = TileDataModule(cfg, metadata)
        m = CNNInterface(cfg.train_model)
    elif cfg.train_model.model_type == 'ADDMIL':
        dm = DataModule(cfg, metadata)
        m = ADDMILInterface(cfg.train_model)
    else:
        raise NotImplementedError()

    dataset_tag = cfg.common.dataset_tag
    basepath = os.path.basename(os.getcwd()).replace("-","_")
    exp_name = f'{basepath}-{dataset_tag}-{cfg.train_model.model_type}-{cfg.common.target_label}-fold{fold_idx}'
    #exp_name = f'{cfg.train_model.model_type}-{cfg.common.target_label}-fold{fold_idx}'

    logger = AimLogger(
        experiment=exp_name, 
        train_metric_prefix='train/',
        val_metric_prefix='valid/',
        test_metric_prefix='test/',
        system_tracking_interval=None
        )

    cbs = [plcallbacks.RichProgressBar(), ExitCallback()]

    if cfg.train_model.trainer.early_stopping:
        cbs.append(plcallbacks.EarlyStopping(monitor=m.valid_monitor, mode=m.monitor_mode, min_delta=0.01, patience=cfg.train_model.trainer.early_stopping_patience))

    # if trainer.global_rank > 0
    if not callable(logger.experiment.hash):
        artifact_dir = os.path.join('results', 'checkpoints', exp_name, logger.experiment.hash)

        if cfg.train_model.trainer.early_stopping:
            cbs.append(plcallbacks.ModelCheckpoint(dirpath=artifact_dir, monitor=m.valid_monitor, mode=m.monitor_mode))
    else:
        raise Exception("idk")

    if cfg.train_model.trainer.get('tune_callback', False):
        cbs.append(
            TuneReportCallback({
                'tune_metric': m.valid_monitor
            }, on='validation_end')
        )

    trainer = pl.Trainer(
        max_epochs=cfg.train_model.trainer.max_epochs,
        accelerator=cfg.train_model.trainer.accelerator,
        callbacks=cbs,
        precision=cfg.train_model.trainer.precision,
        num_sanity_val_steps=0,
        gradient_clip_val=cfg.train_model.trainer.gradient_clip_val,
        logger=logger,
        use_distributed_sampler=False,
        devices=1,
        #strategy='fsdp_native',
        accumulate_grad_batches=cfg.train_model.trainer.accumulate_grad_batches,
        #track_grad_norm=2
    )

    trainer.fit(m, dm)
    trainer.test(m, dm.val_dataloader(), ckpt_path='best')
    inference(cfg, trainer, m, artifact_dir)

@ray.remote(num_gpus=0.5, max_calls=1)
def train_one_fold_r(cfg):
    return train_one_fold(cfg)

def inference(cfg, trainer, model, art_dir):
    target_dir = os.path.join(art_dir, 'preds')
    os.makedirs(target_dir, exist_ok=True)
    df = read_metadata(cfg.create_tiles.output.metadata)
    inf_ds = SlideBatchDataset(df, cfg.extract_features.output.features)
    inf_dl = torch.utils.data.DataLoader(inf_ds, num_workers=cfg.common.num_workers, batch_size=1, pin_memory=False)
    preds = trainer.predict(model, inf_dl)
    for idx, row in inf_ds.df.iterrows():
        dd = torch.load(os.path.join(cfg.inference.input.tiles, f'{row.slide_id}.pt'))
        dd['metadata']['model_ckpt'] = cfg.inference.input.model_checkpoint
        dd['pred'] = preds[idx]
        dd['metadata']['class_map'] = model.hparams.cfg.class_map
        dd['metadata']['target_label'] = model.hparams.cfg.target_label
        dd['attention_scores'] = model.pred_to_attention_map(preds[idx])
        torch.save(dd, os.path.join(target_dir, f'{row.slide_id}.pt'))

def main(cfg):
    if cfg.train_model.dataset.fold == 'all':
        tbl = read_metadata(cfg.train_model.input.metadata)
        futures = {}
        for fold in tbl.fold.unique():
            with open_dict(cfg):
                cfg.train_model.dataset.fold = int(fold)
            futures[fold] = train_one_fold_r.remote(cfg)
        for fold in futures.keys():
            ray.get(futures[fold])
    else:
         train_one_fold(cfg)
