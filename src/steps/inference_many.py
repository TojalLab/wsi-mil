import os
import ray
from omegaconf import OmegaConf
#from lib.etc import read_metadata
import sys
import glob
import pandas as pd
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


@ray.remote(num_gpus=0.5, max_calls=1)
def slide_inference_r(cfg, df):
    slide_inference(cfg, df)

def slide_inference(cfg, df):
    print(cfg)

    #if 'CLAM' in cfg.train_model.model_type:
    model = CLAMInterface.load_from_checkpoint(cfg.inference.input.model_checkpoint).eval()

    inf_ds = SlideBatchDataset(df, cfg.inference.input.features)
    inf_dl = torch.utils.data.DataLoader(inf_ds, num_workers=cfg.common.num_workers, batch_size=1, pin_memory=False)

    os.makedirs(cfg.inference.output.preds, exist_ok=True)

    cbs = [pl.callbacks.RichProgressBar()]
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=False,
        devices=1,
        callbacks=cbs
    ) 
    preds = trainer.predict(model, inf_dl)

    for idx, row in inf_ds.df.iterrows():
        source_path = os.path.join(cfg.inference.input.tiles, f'{row.slide_id}.pt')
        if os.path.exists(source_path):
            dd = torch.load(source_path)
        else:
            dd = {'metadata': {}}
        dd['metadata']['model_ckpt'] = cfg.inference.input.model_checkpoint
        dd['pred'] = preds[idx]
        dd['metadata']['class_map'] = model.hparams.cfg.class_map
        dd['metadata']['target_label'] = model.hparams.cfg.target_label
        dd['attention_scores'] = model.pred_to_attention_map(preds[idx])
        torch.save(dd, os.path.join(cfg.inference.output.preds, f'{row.slide_id}.pt'))
        print(f"Saved to {os.path.join(cfg.inference.output.preds, f'{row.slide_id}.pt')}")


def is_debug_mode():
    gettrace = getattr(sys, 'gettrace', None) 
    return not (gettrace is None or gettrace() is None)

def get_model_fold_checkpoint(cfg, model, fold):
    """
    Returns the model checkpoint file from the model and fold.
    """
    base = f"{cfg.common.results_dir}/checkpoints"
    model_folder_name = f"*-{cfg.common.dataset_tag}-CLAM_MB-{model}-fold{fold}"
    print(f"{base}/{model_folder_name}/*")
    chkp_folder = glob.glob(f"{base}/{model_folder_name}/*")[0]
    model_checkpoint = glob.glob(f"{chkp_folder}/*.ckpt")[0]
    return model_checkpoint, chkp_folder

def main(cfg):
    #print(sys.argv)
    if not is_debug_mode():
        ray.init(num_cpus=60)

    futures, titles = [], {}
    models = cfg.train_many.models
    folds = list(range(cfg.splits.n_folds))

    for model in models:
        for fold in folds:
            ckpt_file, ckpt_folder = get_model_fold_checkpoint(cfg, model, fold)
            model_folder = f"wsimil-{cfg.common.dataset_tag}-CLAM_MB-{model}-fold{fold}"

            c1 = OmegaConf.load(sys.argv[2])
            c2 = OmegaConf.from_dotlist([
                f'common.target_label={model}',
                f'inference.input.model_checkpoint={ckpt_file}',
                f'inference.output.preds={ckpt_folder}/preds',
            ])
            cfg = OmegaConf.merge(c1, c2)
            OmegaConf.resolve(cfg)

            df = pd.read_csv(cfg.inference.input.metadata)

            # inference 
            if not is_debug_mode():
                remote_id = slide_inference_r.remote(cfg, df)
                futures.append(remote_id)
                titles[remote_id] = f"{model} - fold {fold}"
            else:
                slide_inference(cfg, df)

    with create_progress_ctx() as progress:
        task = progress.add_task("Running inference", total=len(futures))
        
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            progress.update(task, description=f"Last done: {titles[done[0]]}", advance=1, refresh=True)

