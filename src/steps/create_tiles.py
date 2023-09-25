import glob
import os
import PIL.Image
import openslide
import pandas as pd
import sys
import rich.progress
import torchvision
import torch

from lib import bkg_detection, brca_tumor_only, brca_tuinv_only
from lib import etc
from lib import plot

import ray

ray.init()

@ray.remote(num_gpus=0.5, num_cpus=8, max_calls=1) # pyright: ignore
def do_bkg_detect(row, cfg, out_fp, target_mpp, orig_mpp):

    if cfg.create_tiles.remove_tiles == '9tissues':
        obj = bkg_detection.BkgDetection(
            bkg_detection.TiledSlide(row.slide_path,
                tile_size=cfg.create_tiles.tile_size,
                step_size=cfg.create_tiles.step_size,
                target_mpp=target_mpp,
                orig_mpp=orig_mpp))
    elif cfg.create_tiles.remove_tiles == 'BRCATU':
        obj = brca_tumor_only.TODetection(
            brca_tumor_only.TiledSlide(row.slide_path,
                tile_size=cfg.create_tiles.tile_size,
                step_size=cfg.create_tiles.step_size,
                target_mpp=target_mpp,
                orig_mpp=orig_mpp))
    elif cfg.create_tiles.remove_tiles == 'TUINV':
        obj = brca_tuinv_only.TODetection(
            brca_tuinv_only.TiledSlide(row.slide_path,
                tile_size=cfg.create_tiles.tile_size,
                step_size=cfg.create_tiles.step_size,
                target_mpp=target_mpp,
                orig_mpp=orig_mpp))
    else:
        raise Exception(f"no method for {cfg.create_tiles.remove_tiles}")

    obj.inference()
    
    tn_size = (2000,1600)

    tn = obj.preds_thumbnail(size=tn_size)
    pred_tn = torchvision.transforms.functional.to_pil_image(tn.to(torch.uint8).permute([2,0,1]))
    img_tn = obj.tiled_slide.slide.get_thumbnail(tn_size)

    dst = plot.pil_concat_side_by_side(img_tn, pred_tn)
    dst.save(os.path.join(cfg.create_tiles.output.masks, f'{row.slide_id}.jpg'))

    obj.save_all_preds(os.path.join(cfg.create_tiles.output.masks, f'{row.slide_id}.pt'))
    obj.filter_coords(labels_to_exclude=cfg.create_tiles.remove_tiles_labels_exclude) # remove ADI, BACK
    obj.save_filtered_tiles_pt(out_fp)

    return len(obj.filtered_coords)

def create_all_tiles(row, cfg, out_fp, target_mpp, orig_mpp):
        obj = bkg_detection.TiledSlide(row.slide_path,
            tile_size=cfg.create_tiles.tile_size,
            step_size=cfg.create_tiles.step_size,
            target_mpp=target_mpp,
            orig_mpp=orig_mpp)

        obj.save_all_tiles_pt(out_fp)
        return len(obj.coords)

def main(cfg):

    tbl = etc.read_metadata(cfg.create_tiles.input)
    os.makedirs(cfg.create_tiles.output.tiles, exist_ok=True)

    if cfg.create_tiles.remove_tiles != False:
        os.makedirs(cfg.create_tiles.output.masks, exist_ok=True)

    if cfg.create_tiles.ignore_mpp:
        target_mpp = 1
        orig_mpp = 1
    else:
        target_mpp = cfg.create_tiles.target_mpp
        orig_mpp = None

    if os.path.exists(cfg.create_tiles.output.metadata) and (not cfg.common.rerun_existing_output):
        print(f'output already exists: {cfg.create_tiles.output.metadata}')
        sys.exit(0)

    progress = etc.create_progress_ctx()
    pb_task1 = progress.add_task(description='slides')

    futures = {}
    if cfg.create_tiles.remove_tiles:
        for idx, row in tbl.iterrows():
            out_fp = os.path.join(cfg.create_tiles.output.tiles, f'{row.slide_id}.pt')
            if os.path.exists(out_fp) and (not cfg.common.rerun_existing_output):
                pass
            else:
                futures[idx] = do_bkg_detect.remote(row, cfg, out_fp, target_mpp, orig_mpp)

    num_tiles = []
    with progress:
        for idx, row in progress.track(tbl.iterrows(), total=len(tbl), task_id=pb_task1):
            progress.update(pb_task1, description=row.slide_id)

            if not os.path.exists(row.slide_path):
                raise Exception(f'slide not found: {row.slide_path}')

            out_fp = os.path.join(cfg.create_tiles.output.tiles, f'{row.slide_id}.pt')

            if os.path.exists(out_fp) and (not cfg.common.rerun_existing_output):
                x = torch.load(out_fp)
                num_tiles.append(len(x['coords']))

            elif cfg.create_tiles.remove_tiles:
                #num_tiles.append(do_bkg_detect(row, cfg, out_fp, target_mpp, orig_mpp))
                num_tiles.append(ray.get(futures[idx]))

            else:
                num_tiles.append(create_all_tiles(row, cfg, out_fp, target_mpp, orig_mpp))

    tbl['num_tiles'] = num_tiles

    filtered_tbl = tbl[tbl.num_tiles >= cfg.create_tiles.min_tiles].reset_index(drop=True)
    n_filtered = len(tbl) - len(filtered_tbl)
    print(f' {n_filtered} slides were filtered out, remaining: ({len(filtered_tbl)} / {len(tbl)})')

    filtered_tbl.to_csv(cfg.create_tiles.output.metadata, index=False)
