from functools import partial

import torch
import torch.nn as nn
import torchmetrics

from architectures.glimpse_mae import BaseGlimpseMae
from architectures.selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from config import NUM_SEG_CLASSES


class SegmentationMae(BaseGlimpseMae):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, out_chans=NUM_SEG_CLASSES)
        self.mae.decoder_pred = nn.Linear(512, 16 ** 2 * NUM_SEG_CLASSES, bias=True)

        self.define_metric('mAP', partial(torchmetrics.classification.MulticlassAccuracy, num_classes=NUM_SEG_CLASSES,
                                          average='macro',
                                          multidim_average='global'))

    def do_metrics(self, mode, out, batch):
        super().do_metrics(mode, out, batch)
        with torch.no_grad():
            segmentation = self.mae.segmentation_output(out['out'])
            self.log_metric(mode, 'mAP', segmentation, batch[1])

    def calculate_loss_one(self, pred, mask, batch):
        return self.mae.forward_seg_loss(pred, batch[1], mask if self.masked_loss else None)


class RandomSegMae(SegmentationMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=RandomGlimpseSelector)


class CheckerboardSegMae(SegmentationMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=CheckerboardGlimpseSelector)


class AttentionSegMae(SegmentationMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=AttentionGlimpseSelector)
