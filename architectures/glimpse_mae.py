import abc
import argparse
import sys
from abc import ABC
from typing import Any, Dict

import torch

from architectures.base import BaseArchitecture
from architectures.mae import mae_vit_large_patch16
from architectures.utils import dict_to_cpu
from datasets.base import BaseDataModule
from datasets.segmentation import BaseSegmentationDataModule


class BaseGlimpseMae(BaseArchitecture, ABC):
    glimpse_selector_class = None

    def __init__(self, args: Any, datamodule: BaseDataModule, out_chans=3):
        super().__init__(args, datamodule)

        self.mae = mae_vit_large_patch16(img_size=datamodule.image_size, out_chans=out_chans)

        self.num_glimpses = args.num_glimpses
        self.masked_loss = args.masked_loss
        self.sum_losses = args.sum_losses
        self.single_step = args.single_step

        assert self.glimpse_selector_class is not None
        self.glimpse_selector = self.glimpse_selector_class(self, args)

        if args.pretrained_mae_path:
            print(self.load_pretrained_mae(args.pretrained_mae_path,
                                           segmentation=isinstance(datamodule, BaseSegmentationDataModule)),
                  file=sys.stderr)

        self.debug = False

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(BaseGlimpseMae.__name__)
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
        parser.add_argument('--single-step',
                            help='do all glimpse selections in one step',
                            type=bool,
                            default=False,
                            action=argparse.BooleanOptionalAction)
        parent_parser = cls.glimpse_selector_class.add_argparse_args(parent_parser)
        return parent_parser

    def load_pretrained_mae(self, path="architectures/mae_vit_l_128x256.pth", segmentation=False):
        checkpoint = torch.load(path, map_location='cpu')
        if 'model' in checkpoint:
            checkpoint = checkpoint["model"]
            prefix = ''
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
            prefix = 'mae.'
        else:
            raise NotImplemented()
        del checkpoint[prefix + 'pos_embed']
        del checkpoint[prefix + 'decoder_pos_embed']

        if segmentation:
            del checkpoint[prefix + 'decoder_pred.weight']
            del checkpoint[prefix + 'decoder_pred.bias']

        if prefix == '':
            return self.mae.load_state_dict(checkpoint, strict=False)
        else:
            return self.load_state_dict(checkpoint, strict=False)

    @abc.abstractmethod
    def calculate_loss_one(self, out, batch):
        raise NotImplemented()

    def calculate_loss(self, losses, batch):
        return torch.mean(torch.stack(losses))

    def forward_one(self, x, mask_indices, mask) -> Dict[str, torch.Tensor]:
        latent = self.mae.forward_encoder(x, mask_indices)
        out = self.mae.forward_decoder(latent, mask, mask_indices)
        return {
            'out': out,
            'latent': latent,
            'mask': mask
        }

    def forward(self, batch, compute_loss=True):
        x = batch[0]
        mask = torch.full((x.shape[0], self.mae.grid_size[1] * self.mae.grid_size[0]), fill_value=False,
                          device=self.device)
        mask_indices = torch.empty((x.shape[0], 0), dtype=torch.int32, device=self.device)
        loss = 0
        losses = []
        steps = []
        if not self.single_step:
            # zero step (initialize decoder attention weights)
            out = self.forward_one(x, mask_indices, mask)
            if self.debug:
                steps.append(dict_to_cpu(out))
        for i in range(self.num_glimpses):
            mask, mask_indices = self.glimpse_selector(mask, mask_indices, i)
            if self.single_step and i + 1 < self.num_glimpses:
                continue
            out = self.forward_one(x, mask_indices, mask)
            if compute_loss:
                loss = self.calculate_loss_one(out, batch)
                losses.append(loss)
            if self.debug:
                steps.append(dict_to_cpu(out | self.glimpse_selector.debug_info))

        if compute_loss and self.sum_losses:
            loss = self.calculate_loss(losses, batch)

        return out | {"losses": losses, "loss": loss, "steps": steps}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch, compute_loss=False)
