import torch
import torchvision
import os.path
import torchmetrics
import pytorch_lightning as pl
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastai.vision.learner import create_vision_model

from . import model_clam
from lib.plot import figure_to_numpy, confusion_matrix_figure, roc_curves_figure, pr_curves_figure

class CNNInterface(pl.LightningModule):
    valid_monitor = 'valid/loss'
    monitor_mode = 'min'

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.setup_model(cfg)

        ms = torchmetrics.MetricCollection([
            torchmetrics.F1Score(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.CohenKappa(num_classes=cfg.n_classes),
            torchmetrics.MatthewsCorrCoef(num_classes=cfg.n_classes),
        ])

        self.train_metrics = ms.clone(prefix='train_metrics/')
        self.valid_metrics = ms.clone(prefix='valid_metrics/')

        self.test_metrics = torchmetrics.MetricCollection(prefix='test_metrics/', metrics=[
            torchmetrics.Accuracy(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.AUROC(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.CohenKappa(num_classes=cfg.n_classes),
            torchmetrics.F1Score(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.Precision(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.Recall(num_classes=cfg.n_classes, average='macro'),
        ])

        self.test_figs = torchmetrics.MetricCollection([
            torchmetrics.ConfusionMatrix(num_classes=cfg.n_classes),
            torchmetrics.ROC(num_classes=cfg.n_classes),
            torchmetrics.PrecisionRecallCurve(num_classes=cfg.n_classes)
        ])

    def setup_model(self, cfg):
        #self.model = create_cnn_model(torchvision.models.resnet18,  cfg.n_classes)
        self.model = create_vision_model(torchvision.models.convnext_tiny,  cfg.n_classes)
        self.freeze_body()

    def freeze_body(self, do_freeze=True):
        for p in self.model[0].parameters():
            p.requires_grad = not do_freeze
        if do_freeze:
            self.my_opt_params = {
                'max_lr': 0.002,
                'pct_start': 0.99,
                'div_factor': 25,
                'body_frozen': True
            }
        else:
            self.my_opt_params = {
                'max_lr': [0.001/100, 0.001],
                'pct_start': 0.3,
                'div_factor': 5,
                'body_frozen': False
            }
    
    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        if self.my_opt_params['body_frozen'] and self.trainer.current_epoch > 1:
            print('unfreezing body')
            # unfreeze body
            self.freeze_body(False)
            self.trainer.accelerator.setup(self.trainer)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_metrics(y_hat, y)
        self.log('train/loss', loss)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.valid_metrics(y_hat, y)
        self.log('valid/loss', loss)

        return loss
    
    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute())
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.test_metrics.update(y_hat, y)
        self.test_figs.update(y_hat, y)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())

        labels = list(map(lambda i:i[0], sorted(self.cfg.class_map.items(), key=lambda i:i[1])))
        figs_data = self.test_figs.cpu().compute()
        cm = figs_data['MulticlassConfusionMatrix'].numpy()
        fpr, tpr, thres = figs_data['MulticlassROC']
        prec, recall, thres = figs_data['MulticlassPrecisionRecallCurve']

        self.logger.experiment.track(aim.Image(confusion_matrix_figure(cm, labels=labels)), name='cm', context={'subset':'test'}) # pyright: ignore
        self.logger.experiment.track(aim.Image(pr_curves_figure(prec, recall, labels=labels)), name='pr', context={'subset':'test'}) # pyright: ignore
        self.logger.experiment.track(aim.Image(roc_curves_figure(fpr, tpr, labels=labels)), name='roc', context={'subset':'test'}) # pyright: ignore

    # def predict_step(self, batch, batch_idx):
    #     #x, y = batch
    #     return self(x)
    
    def configure_optimizers(self):
        if self.my_opt_params['body_frozen']:
            opt = torch.optim.AdamW(self.model[1].parameters(), 
                betas=(0.9,0.99), eps=1e-5, weight_decay=0.01)
        else:
            opt = torch.optim.AdamW([
                {'params': self.model[0].parameters() },
                {'params': self.model[1].parameters() }],
                betas=(0.9,0.99), eps=1e-5, weight_decay=0.01)
        self.trainer.reset_train_dataloader() # init trainloader
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.my_opt_params['max_lr'],
                div_factor=self.my_opt_params['div_factor'],
                pct_start=self.my_opt_params['pct_start'],
                epochs=(self.trainer.max_epochs-self.trainer.current_epoch),
                steps_per_epoch=len(self.trainer.train_dataloader)
            ),
            "interval": "step"
        }
        return {"optimizer": opt, "lr_scheduler": scheduler_dict}
        