import torch
import torch.utils.data
from lib.etc import create_imagenet_feature_extractor, read_metadata, create_progress_ctx
from lib import bkg_detection
import torchvision
import os
import os.path
import ray
from types import SimpleNamespace
from tqdm import tqdm
from rich.progress import Progress, TimeElapsedColumn


ray.init(num_cpus=60)

@ray.remote(num_gpus=0.12, max_calls=1) # pyright: ignore
def do_extract_feat(cfg, row):
    fp = os.path.join(cfg.extract_features.input.tiles, f'{row.slide_id}.pt')
    fout = os.path.join(cfg.extract_features.output.features, f'{row.slide_id}.pt')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    fe = torchvision.models.resnet50()
    fe.fc = torch.nn.Identity()
    w = torch.load("pretrained_models/RetCCL_resnet50_weights.pth")
    fe.load_state_dict(w, strict=True)
    fe = fe.to(device)
    fe = fe.eval()

    obj = bkg_detection.TiledSlide.from_tiles_pt(fp)
    ds = bkg_detection.TileDataset(obj, cfg.extract_features.input.color_norm)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, 
                                        shuffle=False, 
                                        num_workers=cfg.common.num_workers, 
                                        drop_last=False, pin_memory=False)

    preds = []
    with torch.no_grad():
        for x in dl:
            p = fe(x.to(device)).detach().cpu()
            preds.append(p)

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

    futures, titles = [], {}
    for idx,row in tbl.iterrows():
        fp = os.path.join(cfg.extract_features.input.tiles, f'{row.slide_id}.pt')
        fout = os.path.join(cfg.extract_features.output.features, f'{row.slide_id}.pt')
        if not os.path.exists(fp):
            raise Exception(f'tiles not found: {fp}')
        if os.path.exists(fout) and (not cfg.common.rerun_existing_output):
            continue
        remote_id = do_extract_feat.remote(cfg, row)
        futures.append(remote_id)
        titles[remote_id] = row.slide_id

    with Progress(*Progress.get_default_columns(),TimeElapsedColumn()) as progress:
        features_task = progress.add_task("Extracting features", total=len(futures))
        
        while futures:
            done, futures = ray.wait(futures, num_returns=1)
            progress.update(features_task, description=f"Last finished: {titles[done[0]]}", advance=1, refresh=True)

