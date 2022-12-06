import torch
import torchmetrics

from architectures.glimpse_mae import BaseGlimpseMae
from architectures.selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from datasets.utils import IMAGENET_MEAN, IMAGENET_STD


class ReconstructionMae(BaseGlimpseMae):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_rmse_overall = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse_overall = torchmetrics.MeanSquaredError(squared=False)
        self.train_tina = torchmetrics.MeanMetric()
        self.val_tina = torchmetrics.MeanMetric()
        self.train_rmse_masked = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse_masked = torchmetrics.MeanSquaredError(squared=False)

        self.register_buffer('imagenet_mean', torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1))

    def calculate_loss_one(self, pred, mask, batch):
        return self.mae.forward_loss(batch[0], pred, mask if self.masked_loss else None)

    def __rev_normalize(self, img):
        return torch.clip((img * self.imagenet_std + self.imagenet_mean) * 255, 0, 255)

    def train_log_metrics(self, out, batch):
        with torch.no_grad():
            target = self.__rev_normalize(batch[0])

            reconstructed = self.mae.reconstruct(out['out'], target, out['mask'])
            reconstructed = self.__rev_normalize(reconstructed)

            self.log('train/rmse_overall', self.train_rmse_overall(reconstructed, target), on_step=True, on_epoch=True)
            tina_metric = torch.mean(torch.sqrt(torch.sum((reconstructed - target) ** 2, 1)), [0, 1, 2])
            self.log('train/tina', self.train_tina(tina_metric), on_step=True, on_epoch=True)

            mask_neg = ~out['mask']
            self.log('train/rmse_masked',
                     self.train_rmse_masked(self.mae.patchify(reconstructed)[mask_neg, :],
                                            self.mae.patchify(target)[mask_neg, :]), on_step=True, on_epoch=True)

    def val_log_metrics(self, out, batch):
        with torch.no_grad():
            target = self.__rev_normalize(batch[0])

            reconstructed = self.mae.reconstruct(out['out'], target, out['mask'])
            reconstructed = self.__rev_normalize(reconstructed)

            self.log('val/rmse_overall', self.val_rmse_overall(reconstructed, target), on_step=False, on_epoch=True,
                     sync_dist=True)
            tina_metric = torch.mean(torch.sqrt(torch.sum((reconstructed - target) ** 2, 1)), [0, 1, 2])
            self.log('val/tina', self.val_tina(tina_metric), on_step=False, on_epoch=True, sync_dist=True)

            mask_neg = ~out['mask']
            self.log('val/rmse_masked',
                     self.val_rmse_masked(self.mae.patchify(reconstructed)[mask_neg, :],
                                          self.mae.patchify(target)[mask_neg, :]),
                     on_step=False, on_epoch=True, sync_dist=True)


class RandomMae(ReconstructionMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=RandomGlimpseSelector())


class CheckerboardMae(ReconstructionMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=CheckerboardGlimpseSelector())


class AttentionMae(ReconstructionMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=AttentionGlimpseSelector())
