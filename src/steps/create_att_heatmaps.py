import os.path
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image
import rich.progress
import torch
import os
import openslide
import numpy as np
import matplotlib
import pytorch_lightning as pl

from lib.etc import read_metadata, create_progress_ctx
from lib.models.clam_interface import CLAMInterface
from lib.models.dataset_interface import SlideBatchDataset
from lib.models.transmil_interface import TransMILInterface
from lib.plot import pil_concat_side_by_side

def slide_inference(cfg, df):

    if 'CLAM' in cfg.train_model.model_type:
        model = CLAMInterface.load_from_checkpoint(cfg.att_heatmaps.input.model_checkpoint).eval()
    elif cfg.train_model.model_type == 'TransMIL':
        model = TransMILInterface.load_from_checkpoint(cfg.att_heatmaps.input.model_checkpoint).eval()
    else:
        raise NotImplementedError()

    inf_ds = SlideBatchDataset(df, cfg.extract_features.output.features, cfg.common.target_label)
    inf_dl = torch.utils.data.DataLoader(inf_ds, num_workers=cfg.common.num_workers, batch_size=1, pin_memory=False)

    os.makedirs(cfg.att_heatmaps.output.preds, exist_ok=True)

    cbs = [pl.callbacks.RichProgressBar()]
    trainer = pl.Trainer(accelerator=cfg.train_model.trainer.accelerator, logger=False, callbacks=cbs)
    preds = trainer.predict(model, inf_dl)

    for idx, row in inf_ds.df.iterrows():
        dd = torch.load(os.path.join(cfg.att_heatmaps.input.tiles, f'{row.slide_id}.pt'))
        dd['metadata']['model_ckpt'] = cfg.att_heatmaps.input.model_checkpoint
        dd['pred'] = preds[idx]
        dd['attention_scores'] = model.pred_to_attention_map(preds[idx])
        torch.save(dd, os.path.join(cfg.att_heatmaps.output.preds, f'{row.slide_id}.pt'))

def slide_att_heatmap(att_pt_file, tn_size=(2000,1600), gamma=1, cmap='turbo'):

    dd = torch.load(att_pt_file)
    slide = openslide.open_slide(dd['metadata']['slide_path'])
    tn = slide.get_thumbnail(tn_size)

    fct = np.array(tn.size) / np.array(slide.dimensions)
    tile_size = dd['metadata']['tile_size']

    vv = torch.zeros(tn.size)
    cc = torch.zeros(tn.size)
    for i, c in enumerate(dd['coords']):
        x1,y1 = (c * fct).round().int()
        x2,y2 = ((c+tile_size) * fct).round().int()
        vv[x1:x2,y1:y2] += dd['attention_scores'][i]
        cc[x1:x2,y1:y2] += 1
    vv = vv/cc

    clr = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.PowerNorm(gamma=gamma,vmin=vv.nan_to_num().min(),vmax=vv.nan_to_num().max()),
        cmap=cmap)
    ps = clr.to_rgba(vv.T, bytes=True)

    heatmap = PIL.Image.fromarray(ps).convert('RGB')

    return tn, heatmap

def main(cfg):
    df = pd.read_csv(cfg.att_heatmaps.input.metadata)
    df = df[df.is_valid==1]
    df = df[df.fold==0]

    # inference 
    slide_inference(cfg, df)

    # heatmaps
    os.makedirs(cfg.att_heatmaps.output.overlay, exist_ok=True)
    os.makedirs(cfg.att_heatmaps.output.side_by_side, exist_ok=True)
    progress = create_progress_ctx()
    pb_task1 = progress.add_task(description='heatmaps')
    with progress:
        for idx, row in progress.track(df.iterrows(), total=len(df), task_id=pb_task1):

            inf_pt = os.path.join(cfg.att_heatmaps.output.preds, f'{row.slide_id}.pt')
            tn, heatmap = slide_att_heatmap(inf_pt, 
                                tn_size=cfg.att_heatmaps.thumbnail_size,
                                gamma=cfg.att_heatmaps.gamma,
                                cmap=cfg.att_heatmaps.cmap)

            dst = pil_concat_side_by_side(tn, heatmap)
            dst.save(os.path.join(cfg.att_heatmaps.output.side_by_side, f'{row.slide_id}.jpg'))
            PIL.Image.blend(tn, heatmap, alpha=cfg.att_heatmaps.alpha_blend).save(os.path.join(cfg.att_heatmaps.output.overlay, f'{row.slide_id}.jpg'))
