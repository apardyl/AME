import abc
import argparse
from abc import ABC
from typing import Optional, Any, Callable

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch.optim import AdamW

from architectures.mae import mae_vit_large_patch16
from architectures.utils import MaeScheduler
from datasets.base import BaseDataModule


class BaseGlimpseMae(LightningModule, ABC):
    def __init__(self, args: Any, datamodule: BaseDataModule, glimpse_selector=None, out_chans=3):
        super().__init__()

        self.mae = mae_vit_large_patch16(img_size=datamodule.image_size, out_chans=out_chans)

        self.num_glimpses = args.num_glimpses
        self.lr = args.lr
        self.min_lr = args.min_lr
        self.warmup_epochs = args.warmup_epochs
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.masked_loss = args.masked_loss
        self.glimpse_size = args.glimpse_size
        self.glimpse_selector = glimpse_selector(self)

        self.current_lr = args.lr

        self.save_hyperparameters(ignore=['glimpse_selector'])

        self.define_metric('loss', torchmetrics.MeanMetric)

    def define_metric(self, name: str, metric_constructor: Callable):
        for mode in ['train', 'val', 'test']:
            setattr(self, f'{mode}_{name}', metric_constructor())

    def log_metric(self, mode: str, name: str, *args, on_step: bool = False, on_epoch: bool = True,
                   sync_dist: bool = True, prog_bar: bool = False, **kwargs) -> None:
        self.log(name=f'{mode}/{name}', value=getattr(self, f'{mode}_{name}')(*args, **kwargs), on_step=on_step,
                 on_epoch=on_epoch, sync_dist=sync_dist, prog_bar=prog_bar)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(BaseGlimpseMae.__name__)
        parser.add_argument('--lr',
                            help='learning-rate',
                            type=float,
                            default=1.5e-4)
        parser.add_argument('--warmup-epochs',
                            help='epochs to warmup LR',
                            type=int,
                            default=10)
        parser.add_argument('--min-lr',
                            help='lower lr bound for cyclic schedulers that hit 0',
                            type=float,
                            default=1e-8)
        parser.add_argument('--weight-decay',
                            help='weight_decay',
                            type=float,
                            default=0)
        parser.add_argument('--num-glimpses',
                            help='number of glimpses to take',
                            type=int,
                            default=8)
        parser.add_argument('--glimpse-size',
                            help='size of a glimpse (in number of patches)',
                            type=int,
                            default=3)
        parser.add_argument('--masked-loss',
                            help='calculate loss only for masked patches',
                            type=bool,
                            default=True,
                            action=argparse.BooleanOptionalAction)
        return parent_parser

    def load_pretrained_mae(self, path="architectures/mae_vit_l_128x256.pth", segmentation=False):
        checkpoint = torch.load(path, map_location='cpu')["model"]
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']

        if segmentation:
            del checkpoint['decoder_pred.weight']
            del checkpoint['decoder_pred.bias']

        return self.mae.load_state_dict(checkpoint, strict=False)

    @abc.abstractmethod
    def calculate_loss_one(self, pred, mask, batch):
        raise NotImplemented()

    def calculate_loss(self, losses, batch):
        loss = torch.mean(torch.stack(losses))
        return loss

    def forward_one(self, x, mask_indices, mask):
        latent = self.mae.forward_encoder(x, mask_indices)
        pred = self.mae.forward_decoder(latent, mask, mask_indices)
        return pred

    def forward(self, batch, compute_loss=True):
        x = batch[0]
        mask = torch.full((x.shape[0], self.mae.grid_size[1] * self.mae.grid_size[0]), fill_value=False,
                          device=self.device)
        mask_indices = torch.empty((x.shape[0], 0), dtype=torch.int32, device=self.device)
        losses = []
        loss = 0
        with torch.no_grad():
            latent = self.mae.forward_encoder(x, mask_indices)
            pred = self.mae.forward_decoder(latent, mask, mask_indices)
        for i in range(self.num_glimpses):
            mask, mask_indices = self.glimpse_selector(mask, mask_indices, i)
            pred = self.forward_one(x, mask_indices, mask)
            if compute_loss:
                losses.append(self.calculate_loss_one(pred, mask, batch))
        if compute_loss:
            loss = self.calculate_loss(losses, batch)
        return {"out": pred, "mask": mask, "losses": losses, "loss": loss}

    def do_metrics(self, mode, out, batch):
        pass

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = out['loss']
        self.log_metric('train', 'loss', loss, on_step=True, on_epoch=False, sync_dist=False)
        self.log('train/lr', self.current_lr, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.do_metrics('train', out, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.log_metric('val', 'loss', out['loss'])
        self.do_metrics('val', out, batch)

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.log_metric('test', 'loss', out['loss'])
        self.do_metrics('test', out, batch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.95))
        scheduler = MaeScheduler(optimizer=optimizer,
                                 lr=self.lr,
                                 warmup_epochs=self.warmup_epochs,
                                 min_lr=self.min_lr,
                                 epochs=self.epochs)
        scheduler.step(epoch=0)

        lr_schedulers = [
            {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        ]

        return [optimizer], lr_schedulers

    def lr_scheduler_step(self, scheduler, optimizer_idx: int, metric: Optional[Any]) -> None:
        self.current_lr = scheduler.step(epoch=self.current_epoch + 1)
