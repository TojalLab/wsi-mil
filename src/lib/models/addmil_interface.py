import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import ConfusionMatrixDisplay
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import matplotlib
from timm.optim import create_optimizer

import aim

from . import model_addmil
from lib.plot import figure_to_numpy, confusion_matrix_figure, roc_curves_figure, pr_curves_figure

class ADDMILInterface(pl.LightningModule):

    valid_monitor = 'valid/loss'
    monitor_mode = 'min'

    #---->init
    def __init__(self, cfg):
        super(ADDMILInterface, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = model_addmil.ADDMIL(num_classes=cfg.n_classes, num_features=1280)
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=cfg.model.label_smoothing)
        self.optimizer = {
            'opt':'adam',
            'lr': 2e-4
        }

        ms = torchmetrics.MetricCollection([
            torchmetrics.F1Score(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.CohenKappa(num_classes=cfg.n_classes),
            torchmetrics.MatthewsCorrCoef(num_classes=cfg.n_classes),
        ])

        self.train_metrics = ms.clone(prefix='train/')
        self.valid_metrics = ms.clone(prefix='valid/')

        self.test_metrics = torchmetrics.MetricCollection(prefix='test/', metrics=[
            torchmetrics.Accuracy(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.AUROC(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.CohenKappa(num_classes=cfg.n_classes),
            torchmetrics.F1Score(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.Precision(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.Recall(num_classes=cfg.n_classes, average='macro')
        ])

        self.test_figs = torchmetrics.MetricCollection([
            torchmetrics.ConfusionMatrix(num_classes=cfg.n_classes),
            torchmetrics.ROC(num_classes=cfg.n_classes),
            torchmetrics.PrecisionRecallCurve(num_classes=cfg.n_classes)
        ])

    def training_step(self, batch, batch_idx):
        #---->inference
        data, label = batch
        #print(data.shape)
        out, a = self.model(data)

        #---->loss
        loss = self.loss(out, label.squeeze())
        self.log('train/loss', loss)
        self.train_metrics(out.unsqueeze(0), label)

        return {'loss': loss}

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())

    def validation_step(self, batch, batch_idx):
        data, label = batch
        out, a = self.model(data)

        loss = self.loss(out, label.squeeze())
        self.log('valid/loss', loss)
        self.valid_metrics(out.unsqueeze(0), label)

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr=self.cfg.CLAM.learning_rate, weight_decay=self.cfg.CLAM.weight_decay)
        return optimizer

    def test_step(self, batch, batch_idx):
        data, label = batch
        out, a = self.model(data)

        self.test_metrics.update(out.unsqueeze(0), label)
        self.test_figs.update(out.unsqueeze(0), label)

        return {'out' : out, 'label' : label}

    def predict_step(self, batch, batch_idx):
        data, label = batch
        return self.model(data)

    @staticmethod
    def pred_to_attention_map(pred, n_head=0):
        return pred['attn'][0,n_head].min(dim=1).values

    def test_epoch_end(self, output_results):
        self.log_dict(self.test_metrics.compute())

        labels = list(map(lambda i:i[0], sorted(self.cfg.class_map.items(), key=lambda i:i[1])))
        figs_data = self.test_figs.cpu().compute()
        cm = figs_data['ConfusionMatrix'].numpy()
        fpr, tpr, thres = figs_data['ROC']
        prec, recall, thres = figs_data['PrecisionRecallCurve']

        self.logger.experiment.track(aim.Image(confusion_matrix_figure(cm, labels=labels)), name='cm', context={'subset':'test'}) # pyright: ignore
        self.logger.experiment.track(aim.Image(pr_curves_figure(prec, recall, labels=labels)), name='pr', context={'subset':'test'}) # pyright: ignore
        self.logger.experiment.track(aim.Image(roc_curves_figure(fpr, tpr, labels=labels)), name='roc', context={'subset':'test'}) # pyright: ignore
