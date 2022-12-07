import abc
import argparse
from abc import ABC
from typing import Optional, Any

import torch
import torchmetrics
from pytorch_lightning import LightningModule
from torch.optim import AdamW

from architectures.mae import mae_vit_large_patch16
from architectures.utils import MaeScheduler
from config import GLIMPSES_W, GLIMPSES_H, IMG_SIZE


class BaseGlimpseMae(LightningModule, ABC):
    def __init__(self, args: Any, glimpse_selector=None, out_chans=3):
        super().__init__()

        self.mae = mae_vit_large_patch16(img_size=IMG_SIZE, out_chans=out_chans)

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        self.num_glimpses = args.num_glimpses
        self.lr = args.lr
        self.min_lr = args.min_lr
        self.warmup_epochs = args.warmup_epochs
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.masked_loss = args.masked_loss
        self.glimpse_selector = glimpse_selector

        self.current_lr = args.lr

        self.save_hyperparameters(ignore=['glimpse_selector'])

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument('--lr',
                            help='learning_rate',
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
        parser.add_argument('--masked_loss',
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
        mask = torch.full((x.shape[0], GLIMPSES_W * GLIMPSES_H), fill_value=False, device=self.device)
        mask_indices = torch.empty((x.shape[0], 0), dtype=torch.int32, device=self.device)
        losses = []
        loss = 0
        with torch.no_grad():
            latent = self.mae.forward_encoder(x, mask_indices)
            pred = self.mae.forward_decoder(latent, mask, mask_indices)
        for i in range(self.num_glimpses):
            mask, mask_indices = self.glimpse_selector(self.mae, mask, mask_indices, i)
            pred = self.forward_one(x, mask_indices, mask)
            if compute_loss:
                losses.append(self.calculate_loss_one(pred, mask, batch))
        if compute_loss:
            loss = self.calculate_loss(losses, batch)
        return {"out": pred, "mask": mask, "losses": losses, "loss": loss}

    def train_log_metrics(self, out, batch):
        pass

    def val_log_metrics(self, out, batch):
        pass

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = out['loss']
        self.log('train/loss', self.train_loss(loss), on_step=True)
        self.log('train/lr', self.current_lr, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.train_log_metrics(out, batch)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.log('val/loss', self.val_loss(out['loss']), on_step=False, on_epoch=True, sync_dist=True)
        self.val_log_metrics(out, batch)

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
