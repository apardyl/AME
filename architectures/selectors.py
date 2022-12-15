import abc

import torch
import torch.nn as nn

from architectures.glimpse_mae import BaseGlimpseMae


class BaseGlimpseSelector:
    def __init__(self, model: BaseGlimpseMae, args):
        self.model = model
        self.glimpse_size = args.glimpse_size
        self.grid_h = model.mae.grid_size[0]
        self.grid_w = model.mae.grid_size[1]

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(BaseGlimpseSelector.__name__)
        parser.add_argument('--glimpse-size',
                            help='size of a glimpse (in number of patches)',
                            type=int,
                            default=3)

        return parent_parser

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, current_mask, mask_indices, glimpse_num):
        raise NotImplemented()


class RandomGlimpseSelector(BaseGlimpseSelector):

    def forward(self, mask, mask_indices, glimpse_num):
        """True in mask represents patches kept, False removed"""
        N, L = mask.shape
        new_glimpse_x = torch.randint(0, self.grid_w - 2, size=(N, 1), device=mask.device)
        new_glimpse_y = torch.randint(0, self.grid_h - 2, size=(N, 1), device=mask.device)
        glimpses = self.grid_w * new_glimpse_y + new_glimpse_x
        glimpses = glimpses.repeat(1, self.glimpse_size)
        for idx in range(1, self.glimpse_size):
            glimpses[:, idx] += idx
        glimpses = torch.cat([glimpses + self.grid_w * idx for idx in range(self.glimpse_size)], dim=1)
        mask = mask.scatter(1, glimpses, torch.full_like(mask, fill_value=True))
        mask_indices = torch.cat((mask_indices, glimpses), dim=1)
        return mask, mask_indices


class CheckerboardGlimpseSelector(BaseGlimpseSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinates = [(1, 1), (5, 1), (9, 1), (13, 1),
                            (1, 5), (5, 5), (9, 5), (13, 5)]

    def forward(self, mask, mask_indices, glimpse_num):
        N, L = mask.shape
        to_take = self.coordinates[glimpse_num]
        new_glimpse_x = torch.full((N, 1), fill_value=to_take[0], device=mask.device)
        new_glimpse_y = torch.full((N, 1), fill_value=to_take[1], device=mask.device)
        glimpses = self.grid_w * new_glimpse_y + new_glimpse_x
        glimpses = glimpses.repeat(1, self.glimpse_size)
        for idx in range(1, self.glimpse_size):
            glimpses[:, idx] += idx
        glimpses = torch.cat([glimpses + self.grid_w * idx for idx in range(self.glimpse_size)], dim=1)
        mask.scatter_(1, glimpses, torch.full_like(mask, fill_value=True))
        mask_indices = torch.cat((mask_indices, glimpses), dim=1)

        return mask, mask_indices


class AttentionGlimpseSelector(BaseGlimpseSelector):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.attention_layer = args.attention_layer

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super().add_argparse_args(parent_parser)
        parser = parent_parser.add_argument_group(AttentionGlimpseSelector.__name__)
        parser.add_argument('--attention-layer',
                            help='number of attention layer in MAE to source entropy information from (0-7)',
                            type=int,
                            default=7)

        return parent_parser

    def forward(self, current_mask, mask_indices, glimpse_num):
        with torch.no_grad():
            current_mask = ~current_mask
            # get self attention weights
            attn = self.model.mae.last_attn[self.attention_layer][..., 1:, 1:]
            B = attn.shape[0]

            # set attention weights to known patches to 0
            attn = attn * current_mask.reshape((B, 1, -1, 1))

            # calculate entropy of attention weights for each output patch
            entropy = (-attn * torch.log2(attn))
            del attn
            entropy = torch.nan_to_num(entropy, nan=0)
            entropy = entropy.sum((1, 3))

            # calculate sampling weights for next glimpse

            entropy = entropy.reshape(shape=(B, 1, self.grid_h, self.grid_w))
            next_mask = nn.functional.avg_pool2d(entropy, self.glimpse_size, 1, (self.glimpse_size - 1) // 2)
            next_mask = nn.functional.pad(next_mask,
                                          (0, self.grid_w - next_mask.shape[3], 0, self.grid_h - next_mask.shape[2]),
                                          mode='constant', value=0)
            assert next_mask.shape == (B, 1, self.grid_h, self.grid_w)
            del entropy

            next_mask[:, :, [0, -1]] = -1e8
            next_mask[:, :, :, [0, -1]] = -1e8
            next_mask = next_mask.reshape(B, -1)
            # select next glimpse
            arg = torch.argmax(next_mask, dim=1).unsqueeze(1)
            next_mask[:] = 0
            next_mask.scatter_(1, arg, next_mask + 1)
            next_mask = next_mask.reshape(B, 1, self.grid_h, self.grid_w)
            next_mask = nn.functional.max_pool2d(next_mask, self.glimpse_size, 1, (self.glimpse_size - 1) // 2)
            next_mask = nn.functional.pad(next_mask,
                                          (0, self.grid_w - next_mask.shape[3], 0, self.grid_h - next_mask.shape[2]),
                                          mode='constant', value=0)
            assert next_mask.shape == (B, 1, self.grid_h, self.grid_w)
            next_mask = next_mask <= 0
            next_mask = next_mask.reshape(B, -1)

            # Mask to indices
            cumsum = torch.cumsum(torch.ones_like(next_mask), dim=1) - 1
            new_indices = cumsum[~next_mask].reshape(B, -1)
            mask_indices = torch.cat((mask_indices, new_indices), dim=1)
            # add next glimpse to current mask
            next_mask = next_mask * current_mask
        return ~next_mask, mask_indices
