from functools import partial

import torch
import torchmetrics

from architectures.glimpse_mae import BaseGlimpseMae
from architectures.selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from datasets.utils import IMAGENET_MEAN, IMAGENET_STD


class ReconstructionMae(BaseGlimpseMae):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('imagenet_mean', torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1))

        self.define_metric('rmse_overall', partial(torchmetrics.MeanSquaredError, squared=False))
        self.define_metric('rmse_pred', partial(torchmetrics.MeanSquaredError, squared=False))
        self.define_metric('rmse_masked', partial(torchmetrics.MeanSquaredError, squared=False))
        self.define_metric('tina', torchmetrics.MeanMetric)

    def calculate_loss_one(self, reconstruction, aux, mask, batch):
        return self.mae.forward_loss(batch[0], reconstruction, mask if self.masked_loss else None)

    def __rev_normalize(self, img):
        return torch.clip((img * self.imagenet_std + self.imagenet_mean) * 255, 0, 255)

    def do_metrics(self, mode, out, batch):
        super().do_metrics(mode, out, batch)
        with torch.no_grad():
            reconstructed = self.mae.reconstruct(out['out'], batch[0], out['mask'])
            reconstructed = self.__rev_normalize(reconstructed)
            pred = self.mae.unpatchify(out['out'])
            pred = self.__rev_normalize(pred)
            target = self.__rev_normalize(batch[0])
            mask_neg = ~out['mask']

            self.log_metric(mode, 'rmse_overall', reconstructed, target)
            self.log_metric(mode, 'rmse_pred', pred, target)
            self.log_metric(mode, 'rmse_masked', self.mae.patchify(reconstructed)[mask_neg, :],
                            self.mae.patchify(target)[mask_neg, :])
            tina_metric = torch.mean(torch.sqrt(torch.sum((pred - target) ** 2, 1)), [0, 1, 2])
            self.log_metric(mode, 'tina', tina_metric)


class RandomMae(ReconstructionMae):
    glimpse_selector_class = RandomGlimpseSelector


class CheckerboardMae(ReconstructionMae):
    glimpse_selector_class = CheckerboardGlimpseSelector


class AttentionMae(ReconstructionMae):
    glimpse_selector_class = AttentionGlimpseSelector
