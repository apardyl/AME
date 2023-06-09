import sys
from functools import partial

import torch
import torch.nn as nn
import torchmetrics

from architectures.glimpse_mae import BaseGlimpseMae
from architectures.selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from datasets.segmentation import BaseSegmentationDataModule
from architectures.mae_uper import mae_vit_large_patch16_dec512d8b_uper


class SegmentationMae(BaseGlimpseMae):
    def __init__(self, args, datamodule):
        super().__init__(args, datamodule, out_chans=datamodule.num_classes)
        assert isinstance(datamodule, BaseSegmentationDataModule)
        self.num_classes = datamodule.num_classes
        self.ignore_label = datamodule.ignore_label
        self.define_metric('mPA', partial(torchmetrics.classification.MulticlassAccuracy, num_classes=self.num_classes,
                                          ignore_index=self.ignore_label,
                                          average='macro',
                                          multidim_average='global'))
        self.define_metric('PA', partial(torchmetrics.classification.MulticlassAccuracy, num_classes=self.num_classes,
                                         ignore_index=self.ignore_label,
                                         average='micro',
                                         multidim_average='global'))
        self.define_metric('mIoU', partial(torchmetrics.classification.MulticlassJaccardIndex, num_classes=self.num_classes,
                                           ignore_index=self.ignore_label,
                                           average='macro',
                                           multidim_average='global'))

    @torch.no_grad()
    def do_metrics(self, mode, out, batch):
        super().do_metrics(mode, out, batch)
        segmentation = self.mae.segmentation_output(out['out'])
        self.log_metric(mode, 'mPA', segmentation, batch[1])
        self.log_metric(mode, 'PA', segmentation, batch[1])
        self.log_metric(mode, 'mIoU', segmentation, batch[1])

    def log_metric(self, mode: str, name: str, *args, on_step: bool = False, on_epoch: bool = True,
                   sync_dist: bool = True, prog_bar: bool = False, **kwargs) -> None:
        self.get_metric(mode, name)(*args, **kwargs)
        self.log(name=f'{mode}/{name}', value=self.get_metric(mode, name), on_step=on_step,
                 on_epoch=on_epoch, sync_dist=sync_dist, prog_bar=prog_bar)

    def forward_seg_loss(self, pred, target):
        pred = self.mae.unpatchify(pred)
        loss = nn.functional.cross_entropy(pred, target, ignore_index=self.ignore_label)
        return loss

    def calculate_loss_one(self, out, batch):
        return self.forward_seg_loss(out['out'], batch[1])


class RandomSegMae(SegmentationMae):
    glimpse_selector_class = RandomGlimpseSelector


class CheckerboardSegMae(SegmentationMae):
    glimpse_selector_class = CheckerboardGlimpseSelector


class AttentionSegMae(SegmentationMae):
    glimpse_selector_class = AttentionGlimpseSelector


class SegmentationMaeUPer(SegmentationMae):
    def __init__(self, args, datamodule):
        super().__init__(args, datamodule)
        self.mae = mae_vit_large_patch16_dec512d8b_uper(img_size=datamodule.image_size, out_chans=datamodule.num_classes)
        if args.pretrained_mae_path:
            checkpoint = torch.load(args.pretrained_mae_path, map_location='cpu')["model"]
            del checkpoint["decoder_pred.weight"]
            del checkpoint["decoder_pred.bias"]
            new_checkpoint = {}
            for k, v in checkpoint.items():
                if "decoder" not in k and "mask_token" not in k:
                    new_checkpoint[k] = v
            self.mae.load_state_dict(new_checkpoint, strict=False)

    def forward_seg_loss(self, pred, target):
        loss = nn.functional.cross_entropy(pred, target, ignore_index=self.ignore_label)
        return loss


class AttentionSegMaeUPer(SegmentationMaeUPer):
    glimpse_selector_class = AttentionGlimpseSelector
