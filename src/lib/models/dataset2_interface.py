import torch
import torch.utils.data
import openslide
import random
import torchvision
import os.path
import pandas as pd
import pytorch_lightning as pl
from omegaconf import open_dict
from lib.etc import imagenet_stats, create_progress_ctx, RandomRotate90
from lib.etc import ColorNorm, RandomHED

class AugTiledDataset(torch.utils.data.Dataset):
    def __init__(self, df, coords_dir, label_col, tile_sample=1700):
        df = df.reset_index(drop=True)
        self.df = df
        self.coords_dir = coords_dir
        self.label_col = label_col
        self.labels = sorted(df[label_col].unique())
        self.label_map = dict([ (l, i) for (i,l) in enumerate(self.labels)])
        self.tile_sample = tile_sample
        self.n_classes = len(self.labels)
        #self.size = 224
        self.level = 0
        self.coord_noise = 5
        self.tfms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            ColorNorm(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            #torchvision.transforms.ToTensor(),
            RandomRotate90(),
            RandomHED(p=1.),
            torchvision.transforms.Normalize(**imagenet_stats),
        ])

        self.slide_coords = {}
        self.slides = {}
        self.slide_meta = {}
        progress = create_progress_ctx()
        pb_task1 = progress.add_task(description='slides')

        filt_rows = []
        with progress:
            for i, row in progress.track(self.df.iterrows(), total=len(self.df), task_id=pb_task1):
                slide_id = self.df.iloc[i].slide_id
                self.slide_coords[slide_id] = []
                fp = os.path.join(self.coords_dir, f'{slide_id}.pt')
                data = torch.load(fp)
                slide_path = data['metadata']['slide_path']
                self.slides[slide_id] = openslide.open_slide(slide_path)
                self.slide_meta[slide_id] = {
                    'level': data['metadata']['level'],
                    'ts': round(data['metadata']['tile_size'] * data['metadata']['scale'])
                }
                _ = self.slides[slide_id].get_thumbnail((200,200)) # force load openslide
                for j, c in enumerate(data['coords']):
                    if data['top_preds'][j] == 'TUM_INV':
                        self.slide_coords[slide_id].append(c)
                
                if len(self.slide_coords[slide_id]) < self.tile_sample:
                    #print('drop '+slide_id)
                    del self.slides[slide_id]
                    del self.slide_coords[slide_id]
                    del self.slide_meta[slide_id]
                else:
                    filt_rows.append(row)
        print(f'got {len(filt_rows)}')
        self.df = pd.DataFrame(filt_rows)

    def __len__(self):
        return len(self.df)

    def get_single_tile(self, slide_id):
        tile_coord = random.choice(self.slide_coords[slide_id]) + torch.randint(-self.coord_noise, self.coord_noise+1, [2])
        sz = (self.slide_meta[slide_id]['ts'], self.slide_meta[slide_id]['ts'])
        return self.tfms(self.slides[slide_id].read_region(tile_coord.tolist(), level=self.slide_meta[slide_id]['level'], size=sz).convert('RGB'))

    def __getitem__(self, idx):
        slide_id = self.df.iloc[idx].slide_id
        y = self.df.iloc[idx][self.label_col]
        imgs = torch.stack([ self.get_single_tile(slide_id) for i in range(self.tile_sample)])
        return imgs, self.label_map[y]

class TileDataModule(pl.LightningDataModule):
    def __init__(self, cfg, metadata, tile_sample=1700):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata
        self.tile_sample = tile_sample
        self.train_df = metadata[metadata.is_valid==0]
        self.valid_df = metadata[metadata.is_valid==1]
        #print('train-df', self.train_df)

        self.setup()
        # add extra vars as hyperparameters
        with open_dict(cfg):
            cfg.train_model.n_classes = len(self.train_ds.labels)
            cfg.train_model.class_map = self.train_ds.label_map
            cfg.train_model.target_label = cfg.common.target_label

    def setup(self, stage=None):
        self.train_ds = AugTiledDataset(self.train_df, 
            self.cfg.train_model.input.features, self.cfg.common.target_label, self.tile_sample)
        self.valid_ds = AugTiledDataset(self.valid_df,
            self.cfg.train_model.input.features, self.cfg.common.target_label, self.tile_sample)
        assert self.train_ds.label_map == self.valid_ds.label_map
        self.train_sampler = None

        self.train_sampler = None
        if self.cfg.train_model.dataset.weighted_sample == 'oversample':
            #print(self.train_ds.df)
            total_per_class = self.train_ds.df.value_counts(self.cfg.common.target_label)
            class_weights = sum(total_per_class)/total_per_class
            print('class_weights: ',class_weights)
            num_samples = len(self.train_ds.df)
            all_weights = list(map(lambda k: class_weights[k], self.train_ds.df[self.cfg.common.target_label]))
            self.train_sampler = torch.utils.data.WeightedRandomSampler(all_weights, num_samples=num_samples, replacement=True)
        elif self.cfg.train_model.dataset.weighted_sample == 'undersample':
            total_per_class = self.train_ds.df.value_counts(self.cfg.common.target_label)
            class_weights = min(total_per_class)/total_per_class
            print('class_weights: ',class_weights)
            num_samples = min(total_per_class)
            all_weights = list(map(lambda k: class_weights[k], self.train_ds.df[self.cfg.common.target_label]))
            self.train_sampler = torch.utils.data.WeightedRandomSampler(all_weights, num_samples=num_samples, replacement=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            num_workers=self.cfg.common.num_workers,
            batch_size=1,
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            pin_memory=False
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_ds,
            num_workers=self.cfg.common.num_workers,
            batch_size=1,
            pin_memory=False
        )