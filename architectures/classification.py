from functools import partial

import torch
import torchmetrics
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from architectures.glimpse_mae import BaseGlimpseMae
from architectures.selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from datasets.classification import BaseClassificationDataModule


class ClassificationMae(BaseGlimpseMae):

    def __init__(self, args, datamodule):
        super().__init__(args, datamodule)
        assert isinstance(datamodule, BaseClassificationDataModule)
        self.num_classes = datamodule.num_classes

        self.define_metric('accuracy',
                           partial(torchmetrics.classification.MulticlassAccuracy, num_classes=self.num_classes))
        self.define_metric('targets', torchmetrics.CatMetric)
        self.define_metric('probs', torchmetrics.CatMetric)

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, self.num_classes),
        )

    def forward_head(self, latent, reconstruction):
        latent = latent[:, 0, :]  # get cls token
        return self.head(latent)

    def calculate_loss_one(self, reconstruction, aux, mask, batch):
        rec_loss = self.mae.forward_loss(batch[0], reconstruction, mask if self.masked_loss else None)
        cls_loss = nn.functional.cross_entropy(aux, batch[1])
        return rec_loss * 0.01 + cls_loss

    def __rev_normalize(self, img):
        return torch.clip((img * self.imagenet_std + self.imagenet_mean) * 255, 0, 255)

    def do_metrics(self, mode, out, batch):
        super().do_metrics(mode, out, batch)
        with torch.no_grad():
            self.log_metric(mode, 'accuracy', out['aux'], batch[1])
            self.get_metric(mode, 'targets').update(batch[1])
            self.get_metric(mode, 'probs').update(torch.softmax(out['aux'], dim=1))

    def do_conf_matrix(self, mode):
        for logger in self.loggers:
            if not isinstance(logger, WandbLogger):
                continue
            logger.experiment.log({
                'conf_mat': wandb.plot.confusion_matrix(
                    y_true=self.get_metric(mode, 'targets').compute().int().tolist(),
                    probs=self.get_metric(mode, 'probs').compute().cpu(),
                    class_names=[f'{x:02d}' for x in range(self.num_classes)])
            })

    def on_train_epoch_end(self) -> None:
        self.do_conf_matrix('train')
        super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.do_conf_matrix('val')
        super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        self.do_conf_matrix('test')
        super().on_test_epoch_end()


class RandomClsMae(ClassificationMae):
    glimpse_selector_class = RandomGlimpseSelector


class CheckerboardClsMae(ClassificationMae):
    glimpse_selector_class = CheckerboardGlimpseSelector


class AttentionClsMae(ClassificationMae):
    glimpse_selector_class = AttentionGlimpseSelector
