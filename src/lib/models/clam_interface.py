import torch
import os.path
import torchmetrics
import torchmetrics.classification
import pytorch_lightning as pl
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
import random
import aim

from . import model_clam
from lib.plot import figure_to_numpy, confusion_matrix_figure, roc_curves_figure, pr_curves_figure


class CLAMInterface(pl.LightningModule):

    valid_monitor = 'valid/total_loss'
    monitor_mode = 'min'

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.setup_model()
        
        ms = torchmetrics.MetricCollection([
            torchmetrics.classification.MulticlassF1Score(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.classification.MulticlassCohenKappa(num_classes=cfg.n_classes),
            torchmetrics.classification.MulticlassMatthewsCorrCoef(num_classes=cfg.n_classes),
            torchmetrics.classification.MulticlassAccuracy(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.classification.MulticlassAUROC(num_classes=cfg.n_classes, average='macro')
        ])

        ms2 = torchmetrics.MetricCollection([
            torchmetrics.classification.MulticlassF1Score(num_classes=cfg.n_classes, average='macro')
        ])

        self.train_metrics = ms.clone(prefix='train/')
        self.train_inst_metrics = ms2.clone(prefix='train/inst/')
        self.valid_metrics = ms.clone(prefix='valid/')
        self.valid_inst_metrics = ms2.clone(prefix='valid/inst/')

        self.test_metrics = torchmetrics.MetricCollection(prefix='test/', metrics=[
            torchmetrics.classification.MulticlassAccuracy(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.classification.MulticlassAUROC(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.classification.MulticlassCohenKappa(num_classes=cfg.n_classes),
            torchmetrics.classification.MulticlassF1Score(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.classification.MulticlassPrecision(num_classes=cfg.n_classes, average='macro'),
            torchmetrics.classification.MulticlassRecall(num_classes=cfg.n_classes, average='macro'),
        ])

        self.test_figs = torchmetrics.MetricCollection([
            torchmetrics.classification.MulticlassConfusionMatrix(num_classes=cfg.n_classes),
            torchmetrics.classification.MulticlassROC(num_classes=cfg.n_classes),
            torchmetrics.classification.MulticlassPrecisionRecallCurve(num_classes=cfg.n_classes)
        ])

    def forward(self, x, label=None, instance_eval=False):
        logits, Y_prob, Y_hat, A_raw, instance_dict = self.model(x, label=label, instance_eval=instance_eval)
        return { 'logits': logits, 
                'Y_prob': Y_prob, 
                'Y_hat': Y_hat, 
                'A_raw': A_raw, 
                'instance_dict': instance_dict 
        }

    def on_fit_start(self):
        self.model.relocate(self.device)

    def on_predict_start(self):
        self.model.relocate(self.device)

    def training_step(self, batch, batch_idx):
        x, label = batch
        x = x.flatten(0,1) # combine batch_size x N
        r = self(x, label=label, instance_eval=True)

        loss = self.loss_fn(r['logits'], label)
        self.log('train/loss', loss)

        instance_loss = r['instance_dict']['instance_loss']
        self.log('train/inst_loss', instance_loss)

        total_loss = self.cfg.CLAM.bag_weight * loss + (1-self.cfg.CLAM.bag_weight ) * instance_loss
        self.log('train/total_loss', total_loss)

        self.train_metrics(r['Y_prob'], label)

        inst_preds = r['instance_dict']['inst_preds']
        inst_labels = r['instance_dict']['inst_labels']
        self.train_inst_metrics(inst_preds, inst_labels)

        return total_loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.log_dict(self.train_inst_metrics.compute())

    def validation_step(self, batch, batch_idx):
        x, label = batch
        x = x.flatten(0,1) # combine batch_size x N

        r = self(x, label=label, instance_eval=True)

        loss = self.loss_fn(r['logits'], label)
        self.log('valid/loss', loss)

        instance_loss = r['instance_dict']['instance_loss']
        self.log('valid/inst_loss', instance_loss)

        total_loss = self.cfg.CLAM.bag_weight * loss + (1-self.cfg.CLAM.bag_weight ) * instance_loss
        self.log('valid/total_loss', total_loss)

        self.valid_metrics(r['Y_prob'], label)

        inst_preds = r['instance_dict']['inst_preds']
        inst_labels = r['instance_dict']['inst_labels']
        self.valid_inst_metrics(inst_preds, inst_labels)

    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute())
        self.log_dict(self.valid_inst_metrics.compute())

    def test_step(self, batch, batch_idx):
        x, label = batch
        x = x.flatten(0,1)
        r = self(x, label=label, instance_eval=True)

        self.test_metrics.update(r['Y_prob'], label)
        self.test_figs.update(r['Y_prob'], label)

    def predict_step(self, batch, batch_idx):
        x, = batch
        #x = x.flatten(0,1)
        return self(x, instance_eval=False)

    @staticmethod
    def pred_to_attention_map(pred):
        y_hat = pred['Y_hat'].argmax().item()
        return pred['A_raw'][y_hat].detach().cpu()


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

    def configure_optimizers(self):
        if self.cfg.CLAM.optim == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                   lr=self.cfg.CLAM.learning_rate, weight_decay=self.cfg.CLAM.weight_decay)
        elif self.cfg.CLAM.optim == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                  lr=self.cfg.CLAM.learning_rate, momentum=0.9, weight_decay=self.cfg.CLAM.weight_decay)
        elif self.cfg.CLAM.optim == 'adamw':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                   lr=self.cfg.CLAM.learning_rate, weight_decay=self.cfg.CLAM.weight_decay)
        else:
            raise NotImplementedError

        if self.cfg.CLAM.sched == 'plateau':
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=5)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': sched,
                    'monitor': self.valid_monitor
                }
            }
        else:
            return optimizer

    def setup_model(self):

        if self.cfg.CLAM.bag_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            self.loss_fn = SmoothTop1SVM(n_classes = self.cfg.n_classes)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        model_dict = {"dropout": self.cfg.CLAM.dropout, 'n_classes': self.cfg.n_classes}
        if self.cfg.model_type == 'clam' and self.cfg.CLAM.subtyping:
            model_dict.update({'subtyping': True})

        if self.cfg.CLAM.model_size is not None and self.cfg.model_type != 'mil':
            model_dict.update({"size_arg": self.cfg.CLAM.model_size})

        if self.cfg.model_type in ['CLAM_SB', 'CLAM_MB']:
            if self.cfg.CLAM.subtyping:
                model_dict.update({'subtyping': True})

            if self.cfg.CLAM.B > 0:
                model_dict.update({'k_sample': self.cfg.CLAM.B})

            if self.cfg.CLAM.inst_loss == 'svm':
                from topk.svm import SmoothTop1SVM
                instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            else:
                instance_loss_fn = torch.nn.CrossEntropyLoss()

            if self.cfg.model_type =='CLAM_SB':
                self.model = model_clam.CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
            elif self.cfg.model_type == 'CLAM_MB':
                self.model = model_clam.CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
