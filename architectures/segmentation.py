from functools import partial

import torch
import torch.nn as nn
import torchmetrics

from architectures.glimpse_mae import BaseGlimpseMae
from architectures.selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from datasets.segmentation import BaseSegmentationDataModule


class SegmentationMae(BaseGlimpseMae):
    def __init__(self, args, datamodule):
        super().__init__(args, datamodule, out_chans=self.num_classes)
        assert isinstance(datamodule, BaseSegmentationDataModule)
        self.num_classes = datamodule.num_classes
        self.mae.decoder_pred = nn.Linear(512, 16 ** 2 * self.num_classes, bias=True)

        self.define_metric('mAP', partial(torchmetrics.classification.MulticlassAccuracy, num_classes=self.num_classes,
                                          average='macro',
                                          multidim_average='global'))

    def do_metrics(self, mode, out, batch):
        super().do_metrics(mode, out, batch)
        with torch.no_grad():
            segmentation = self.mae.segmentation_output(out['out'])
            self.log_metric(mode, 'mAP', segmentation, batch[1])

    def forward_seg_loss(self, pred, target, mask=None):
        pred = self.mae.unpatchify(pred)
        loss = nn.functional.cross_entropy(pred, target, reduction="none").unsqueeze(1)
        loss = self.mae.patchify(loss)
        if mask is not None:
            mask_neg = ~mask.unsqueeze(2)
            loss = (loss * mask_neg).sum() / mask_neg.sum()  # mean loss on removed patches

        return loss

    def calculate_loss_one(self, reconstruction, aux, mask, batch):
        return self.forward_seg_loss(reconstruction, batch[1], mask if self.masked_loss else None)


class RandomSegMae(SegmentationMae):
    glimpse_selector_class = RandomGlimpseSelector


class CheckerboardSegMae(SegmentationMae):
    glimpse_selector_class = CheckerboardGlimpseSelector


class AttentionSegMae(SegmentationMae):
    glimpse_selector_class = AttentionGlimpseSelector
