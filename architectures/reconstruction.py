import torch
import torchmetrics

from architectures.glimpse_mae import BaseGlimpseMae
from architectures.selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from data_utils.datasets import IMAGENET_STD, IMAGENET_MEAN


class ReconstructionMae(BaseGlimpseMae):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.train_tina = torchmetrics.MeanMetric()
        self.val_tina = torchmetrics.MeanMetric()

        self.register_buffer('imagenet_mean', torch.tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(1))
        self.register_buffer('imagenet_std', torch.tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(1))

    def calculate_loss_one(self, pred, mask, batch):
        return self.mae.forward_loss(batch[0], pred, mask)

    def __rev_normalize(self, img):
        return torch.clip((img * self.imagenet_std + self.imagenet_mean) * 255, 0, 255)

    def train_log_metrics(self, out, batch):
        with torch.no_grad():
            reconstructed = self.mae.reconstruct(out['out'], batch[0], out['mask'])
            reconstructed = self.__rev_normalize(reconstructed)
            target = self.__rev_normalize(batch[0])

            self.log('train/rmse', self.train_rmse(reconstructed, target), on_step=True, on_epoch=True, sync_dist=True)
            tina_metric = torch.mean(torch.sqrt(torch.sum((reconstructed - target) ** 2, 1)), [0, 1, 2])
            self.log('train/tina', self.train_tina(tina_metric), on_step=False, on_epoch=True, sync_dist=True)

    def val_log_metrics(self, out, batch):
        with torch.no_grad():
            reconstructed = self.mae.reconstruct(out['out'], batch[0], out['mask'])
            reconstructed = self.__rev_normalize(reconstructed)
            target = self.__rev_normalize(batch[0])

            self.log('val/rmse', self.val_rmse(reconstructed, target), on_step=False, on_epoch=True, sync_dist=True)
            tina_metric = torch.mean(torch.sqrt(torch.sum((reconstructed - target) ** 2, 1)), [0, 1, 2])
            self.log('val/tina', self.val_tina(tina_metric), on_step=False, on_epoch=True, sync_dist=True)


class RandomMae(ReconstructionMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=RandomGlimpseSelector())


class CheckerboardMae(ReconstructionMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=CheckerboardGlimpseSelector())


class AttentionMae(ReconstructionMae):
    def __init__(self, args):
        super().__init__(args, glimpse_selector=AttentionGlimpseSelector())
