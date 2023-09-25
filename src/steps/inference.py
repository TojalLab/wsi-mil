import os.path
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image
import torch
import os
import openslide
import numpy as np
import matplotlib
import pytorch_lightning as pl

from lib.etc import read_metadata, create_progress_ctx
from lib.models.clam_interface import CLAMInterface
from lib.models.dataset_interface import SlideBatchDataset
from lib.plot import pil_concat_side_by_side

def slide_inference(cfg, df):

    if 'CLAM' in cfg.train_model.model_type:
        model = CLAMInterface.load_from_checkpoint(cfg.inference.input.model_checkpoint).eval()
    else:
        raise NotImplementedError()

    inf_ds = SlideBatchDataset(df, cfg.extract_features.output.features)
    inf_dl = torch.utils.data.DataLoader(inf_ds, num_workers=cfg.common.num_workers, batch_size=1, pin_memory=False)

    os.makedirs(cfg.inference.output.preds, exist_ok=True)

    cbs = [pl.callbacks.RichProgressBar()]
    trainer = pl.Trainer(
        accelerator=cfg.train_model.trainer.accelerator,
        logger=False,
        devices=1,
        callbacks=cbs
    )
    preds = trainer.predict(model, inf_dl)

    for idx, row in inf_ds.df.iterrows():
        dd = torch.load(os.path.join(cfg.inference.input.tiles, f'{row.slide_id}.pt'))
        dd['metadata']['model_ckpt'] = cfg.inference.input.model_checkpoint
        dd['pred'] = preds[idx]
        dd['metadata']['class_map'] = model.hparams.cfg.class_map
        dd['metadata']['target_label'] = model.hparams.cfg.target_label
        dd['attention_scores'] = model.pred_to_attention_map(preds[idx])
        torch.save(dd, os.path.join(cfg.inference.output.preds, f'{row.slide_id}.pt'))

def main(cfg):
    df = pd.read_csv(cfg.inference.input.metadata)

    # inference 
    slide_inference(cfg, df)
