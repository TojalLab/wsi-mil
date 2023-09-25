import torch
import torch.utils.data
import random
import numpy
import os.path
import pytorch_lightning as pl
from omegaconf import open_dict
import sys

class SlideBatchDataset(torch.utils.data.Dataset):
    def __init__(self, df, feat_dir, label_col=None, shuffle_inst=False, downsample=None, dropout=0.):
        df = df.reset_index(drop=True)
        self.df = df
        self.feat_dir = feat_dir
        if label_col is not None:
            self.label_col = label_col
            print(df[label_col].unique(),file=sys.stderr)
            self.labels = sorted(df[label_col].unique())
            self.label_map = dict([ (l, i) for (i,l) in enumerate(self.labels)])
        else:
            self.label_col = None
        
        self.shuffle_inst = shuffle_inst
        self.downsample = downsample
        self.dropout = dropout

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx].slide_id
        
        fp = os.path.join(self.feat_dir, f'{x}.pt')
        feats = torch.load(fp)['features']
        if self.shuffle_inst:
            feats = feats[torch.randperm(feats.shape[0])]
            if self.dropout > 0:
                dout = round(len(feats) * (1-self.dropout))
                feats = feats[:dout]
            if self.downsample is not None:
                feats = feats[:self.downsample]
        else:
            if self.downsample is not None and self.downsample < feats.shape[0]:
                k = random.randint(0, (feats.shape[0]-self.downsample))
                feats = feats[k:(k+self.downsample)]
        if self.label_col is not None:
            y = self.df.iloc[idx][self.label_col]
            return feats, self.label_map[y]
        else:
            return feats

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, metadata):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata
        self.train_df = metadata[metadata.is_valid==0].reset_index(drop=True)
        self.valid_df = metadata[metadata.is_valid==1].reset_index(drop=True)

        self.setup()
        # add extra vars as hyperparameters
        with open_dict(cfg):
            cfg.train_model.n_classes = len(self.train_ds.labels)
            cfg.train_model.class_map = self.train_ds.label_map
            cfg.train_model.target_label = cfg.common.target_label

    def get_class_weights(self):
        total_per_class = self.train_ds.df.value_counts(self.cfg.common.target_label)
        class_weights = min(total_per_class)/total_per_class
        return list(map(lambda k: class_weights[k], self.train_ds.labels))

    def setup(self, stage=None):
        self.train_ds = SlideBatchDataset(
            self.train_df, 
            self.cfg.train_model.input.features, 
            self.cfg.common.target_label,
            shuffle_inst=self.cfg.train_model.dataset.shuffle_inst,
            downsample=self.cfg.train_model.dataset.downsample,
            dropout=self.cfg.train_model.dataset.dropout)

        self.valid_ds = SlideBatchDataset(self.valid_df,
            self.cfg.train_model.input.features, 
            self.cfg.common.target_label)
        assert self.train_ds.label_map == self.valid_ds.label_map

        self.train_sampler = None
        if self.cfg.train_model.dataset.weighted_sample == 'oversample':
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
