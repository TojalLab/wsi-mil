import torch
import torch.utils.data
from lib.etc import create_imagenet_feature_extractor, read_metadata, create_progress_ctx
from lib import bkg_detection
import os
import os.path
import ray

ray.init()

@ray.remote(num_gpus=0.5, max_calls=1) # pyright: ignore
def do_extract_feat(cfg, row):
    fp = os.path.join(cfg.create_tiles.output.tiles, f'{row.slide_id}.pt')
    fout = os.path.join(cfg.extract_features.output.features, f'{row.slide_id}.pt')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fe = create_imagenet_feature_extractor(cfg.extract_features.torchvision_model, cfg.extract_features.feature_layer)
    fe = fe.to(device)

    obj = bkg_detection.TiledSlide.from_tiles_pt(fp)
    ds = bkg_detection.TileDataset(obj)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, 
                                        shuffle=False, 
                                        num_workers=cfg.common.num_workers, 
                                        drop_last=False, pin_memory=True)

    preds = []
    with torch.no_grad():
        for x in dl:
            p = fe(x.to(device)).detach()
            preds.append(p.cpu())

    metadata = torch.load(fp)['metadata']
    metadata['feat_model'] = cfg.extract_features.torchvision_model
    metadata['feat_layer'] = cfg.extract_features.feature_layer

    preds = torch.cat(preds).cpu()
    torch.save({
        'metadata': metadata,
        'features': preds
    }, fout)

def main(cfg):

    tbl = read_metadata(cfg.extract_features.input.metadata)

    progress = create_progress_ctx()
    pb_task1 = progress.add_task(description='slides')

    os.makedirs(cfg.extract_features.output.features, exist_ok=True)

    futures = {}
    for idx,row in tbl.iterrows():
        fp = os.path.join(cfg.create_tiles.output.tiles, f'{row.slide_id}.pt')
        fout = os.path.join(cfg.extract_features.output.features, f'{row.slide_id}.pt')
        if not os.path.exists(fp):
            raise Exception(f'tiles not found: {fp}')
        if os.path.exists(fout) and (not cfg.common.rerun_existing_output):
            continue
        futures[idx] = do_extract_feat.remote(cfg, row)

    with progress:
        for idx in  progress.track(futures.keys(),  task_id=pb_task1, total=len(futures)):
            ray.get(futures[idx])

