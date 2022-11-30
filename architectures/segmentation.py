import torch.nn as nn
import torchmetrics

from architectures.glimpse_mae import BaseGlimpseMae
from architectures.selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from config import NUM_SEG_CLASSES


class SegmentationMae(BaseGlimpseMae):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, out_chans=NUM_SEG_CLASSES)
        self.mae.decoder_pred = nn.Linear(512, 16 ** 2 * NUM_SEG_CLASSES, bias=True)

        self.train_mAP = torchmetrics.AveragePrecision(num_classes=NUM_SEG_CLASSES, average='macro', task='multilabel')
        self.val_mAP = torchmetrics.AveragePrecision(num_classes=NUM_SEG_CLASSES, average='macro', task='multilabel')

    def calculate_loss_one(self, pred, mask, batch):
        return self.mae.forward_seg_loss(pred, batch[1], mask)

    def train_log_metrics(self, out, batch):
        segmentation = self.mae.segmentation_output(out['out'])
        self.log('train/mAP', self.train_mAP(segmentation, batch[1]), on_step=True, on_epoch=True, sync_dist=True)

    def val_log_metrics(self, out, batch):
        segmentation = self.mae.segmentation_output(out['out'])
        self.log('val/mAP', self.val_mAP(segmentation, batch[1]), on_step=True, on_epoch=True, sync_dist=True)


class RandomSegMae(SegmentationMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=RandomGlimpseSelector())


class CheckerboardSegMae(SegmentationMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=CheckerboardGlimpseSelector())


class AttentionSegMae(SegmentationMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=AttentionGlimpseSelector())
