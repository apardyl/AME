from functools import partial

import torch
import torchmetrics
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

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Linear(256, self.num_classes),
        )

    def forward_head(self, latent, reconstruction):
        latent = latent[:, 0, :] # get cls token
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


class RandomClsMae(ClassificationMae):
    glimpse_selector_class = RandomGlimpseSelector


class CheckerboardClsMae(ClassificationMae):
    glimpse_selector_class = CheckerboardGlimpseSelector


class AttentionClsMae(ClassificationMae):
    glimpse_selector_class = AttentionGlimpseSelector
