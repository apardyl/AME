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
from datasets.segmentation import BaseSegmentationDataModule


class BaseGlimpseMae(LightningModule, ABC):
    glimpse_selector_class = None

    def __init__(self, args: Any, datamodule: BaseDataModule, out_chans=3):
        super().__init__()

        self.mae = mae_vit_large_patch16(img_size=datamodule.image_size, out_chans=out_chans)

        self.num_glimpses = args.num_glimpses
        self.lr = args.lr
        self.min_lr = args.min_lr
        self.warmup_epochs = args.warmup_epochs
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.masked_loss = args.masked_loss
        self.sum_losses = args.sum_losses

        assert self.glimpse_selector_class is not None
        self.glimpse_selector = self.glimpse_selector_class(self, args)

        self.current_lr = args.lr

        self.save_hyperparameters(ignore=['datamodule', 'out_chans'])

        self.define_metric('loss', torchmetrics.MeanMetric)

        if args.pretrained_mae_path:
            self.load_pretrained_mae(args.pretrained_mae_path,
                                     segmentation=isinstance(datamodule, BaseSegmentationDataModule))

        self.debug = False

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
        parser.add_argument('--masked-loss',
                            help='calculate loss only for masked patches',
                            type=bool,
                            default=True,
                            action=argparse.BooleanOptionalAction)
        parser.add_argument('--pretrained-mae-path',
                            help='path to pretrained MAE weights',
                            type=str,
                            default='architectures/mae_vit_l_128x256.pth')
        parser.add_argument('--sum-losses',
                            help='sum losses for all steps',
                            type=bool,
                            default=True,
                            action=argparse.BooleanOptionalAction)
        parent_parser = cls.glimpse_selector_class.add_argparse_args(parent_parser)
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
    def calculate_loss_one(self, reconstruction, aux, mask, batch):
        raise NotImplemented()

    def calculate_loss(self, losses, batch):
        loss = torch.mean(torch.stack(losses))
        return loss

    def forward_head(self, reconstruction, cls_token):
        return None

    def forward_one(self, x, mask_indices, mask):
        latent = self.mae.forward_encoder(x, mask_indices)
        reconstruction, cls_token = self.mae.forward_decoder(latent, mask, mask_indices)
        aux = self.forward_head(reconstruction, cls_token)
        return reconstruction, aux

    def forward(self, batch, compute_loss=True):
        x = batch[0]
        mask = torch.full((x.shape[0], self.mae.grid_size[1] * self.mae.grid_size[0]), fill_value=False,
                          device=self.device)
        mask_indices = torch.empty((x.shape[0], 0), dtype=torch.int32, device=self.device)
        loss = 0
        losses = []
        steps = []
        # zero step (initialize decoder attention weights)
        reconstruction, aux = self.forward_one(x, mask_indices, mask)
        if self.debug:
            steps.append(
                (mask.detach().clone().cpu(), reconstruction.detach().clone().cpu(), None))
        for i in range(self.num_glimpses):
            mask, mask_indices = self.glimpse_selector(mask, mask_indices, i)
            reconstruction, aux = self.forward_one(x, mask_indices, mask)
            if compute_loss:
                loss = self.calculate_loss_one(reconstruction, aux, mask, batch)
                losses.append(loss)
            if self.debug:
                steps.append(
                    (mask.detach().clone().cpu(), reconstruction.detach().clone().cpu(),
                     self.glimpse_selector.debug_info))

        if compute_loss and self.sum_losses:
            loss = self.calculate_loss(losses, batch)

        return {"out": reconstruction, "mask": mask, "losses": losses, "loss": loss, "steps": steps, "aux": aux}

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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch, compute_loss=False)

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
