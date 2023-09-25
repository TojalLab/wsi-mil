
import ray
from omegaconf import OmegaConf, open_dict
from .train_model import train_one_fold_r, train_one_fold
from lib.etc import read_metadata
from rich.progress import Progress, TimeElapsedColumn
import sys
import torch
import sys

def is_debug_mode():
    gettrace = getattr(sys, 'gettrace', None) 
    return not (gettrace is None or gettrace() is None)

def main(cfg):
    if not is_debug_mode():
        ray.init(num_cpus=60)
    print(sys.argv)

    futures, titles = [], {}
    models = cfg.train_many.models 
    for model in models:
        c1 = OmegaConf.load(sys.argv[2])
        c2 = OmegaConf.from_dotlist([f'common.target_label={model}'])
        cfg = OmegaConf.merge(c1, c2)
        OmegaConf.resolve(cfg)

        tbl = read_metadata(cfg.train_model.input.metadata)

        for fold in tbl.fold.unique():
            with open_dict(cfg):
                cfg.train_model.dataset.fold = int(fold)
            if is_debug_mode():
                train_one_fold(cfg)
            else:
                remote = train_one_fold_r.remote(cfg)
            futures.append(remote)
            titles[remote] = f"{model} - fold: {fold}"


    with Progress(*Progress.get_default_columns(),TimeElapsedColumn(), speed_estimate_period=1e6) as progress:
        task = progress.add_task("Training models", total=len(futures))
        
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            progress.update(task, description=f"Last finished: {titles[done[0]]}", advance=1, refresh=True)

