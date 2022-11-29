import abc

import torch
import torch.nn as nn

from config import GLIMPSES_W, GLIMPSES_H


class BaseGlimpseSelector(nn.Module):
    @abc.abstractmethod
    def forward(self, mae, current_mask, mask_indices, glimpse_num):
        raise NotImplemented()


class RandomGlimpseSelector(BaseGlimpseSelector):
    def __init__(self):
        super().__init__()

    def forward(self, mae, mask, mask_indices, glimpse_num):
        """True in mask represents patches kept, False removed"""
        N, L = mask.shape
        new_glimpse_x = torch.randint(0, GLIMPSES_W - 2, size=(N, 1), device=mask.device)
        new_glimpse_y = torch.randint(0, GLIMPSES_H - 2, size=(N, 1), device=mask.device)
        glimpses = GLIMPSES_W * new_glimpse_y + new_glimpse_x
        glimpses = glimpses.repeat(1, 3)
        glimpses[:, 1] += 1
        glimpses[:, 2] += 2
        glimpses = torch.cat((glimpses, glimpses + GLIMPSES_W, glimpses + 2 * GLIMPSES_W), dim=1)
        mask = mask.scatter(1, glimpses, torch.full_like(mask, fill_value=True))
        mask_indices = torch.cat((mask_indices, glimpses), dim=1)
        return mask, mask_indices


class CheckerboardGlimpseSelector(BaseGlimpseSelector):
    def __init__(self):
        super().__init__()
        self.coordinates = [(1, 1), (5, 1), (9, 1), (13, 1),
                            (1, 5), (5, 5), (9, 5), (13, 5)]

    def forward(self, mae, mask, mask_indices, glimpse_num):
        N, L = mask.shape
        to_take = self.coordinates[glimpse_num]
        new_glimpse_x = torch.full((N, 1), fill_value=to_take[0], device=mask.device)
        new_glimpse_y = torch.full((N, 1), fill_value=to_take[1], device=mask.device)
        glimpses = GLIMPSES_W * new_glimpse_y + new_glimpse_x
        glimpses = glimpses.repeat(1, 3)
        glimpses[:, 1] += 1
        glimpses[:, 2] += 2
        glimpses = torch.cat((glimpses, glimpses + GLIMPSES_W, glimpses + 2 * GLIMPSES_W), dim=1)
        mask.scatter_(1, glimpses, torch.full_like(mask, fill_value=True))
        mask_indices = torch.cat((mask_indices, glimpses), dim=1)

        return mask, mask_indices


class AttentionGlimpseSelector(BaseGlimpseSelector):
    def __init__(self):
        super().__init__()

    def forward(self, mae, current_mask, mask_indices, glimpse_num):
        with torch.no_grad():
            current_mask = ~current_mask
            # get self attention weights
            attn = mae.last_attn[7][..., 1:, 1:]
            B = attn.shape[0]

            # set attention weights to known patches to 0
            attn = attn * current_mask.reshape((B, 1, -1, 1))

            # calculate entropy of attention weights for each output patch
            entropy = (-attn * torch.log2(attn))
            del attn
            entropy = torch.nan_to_num(entropy, nan=0)
            entropy = entropy.sum((1, 3))

            # calculate sampling weights for next glimpse
            h = mae.grid_size[0]
            w = mae.grid_size[1]
            entropy = entropy.reshape(shape=(B, 1, h, w))
            next_mask = nn.functional.avg_pool2d(entropy, 3, 1, 1)
            del entropy

            next_mask[:, :, [0, -1]] = -1e8
            next_mask[:, :, :, [0, -1]] = -1e8
            next_mask = next_mask.reshape(B, -1)
            # select next glimpse
            arg = torch.argmax(next_mask, dim=1).unsqueeze(1)
            next_mask[:] = 0
            next_mask.scatter_(1, arg, next_mask + 1)
            next_mask = next_mask.reshape(B, 1, h, w)
            next_mask = nn.functional.max_pool2d(next_mask, 3, 1, 1) <= 0
            next_mask = next_mask.reshape(B, -1)

            # Mask to indices
            cumsum = torch.cumsum(torch.ones_like(next_mask), dim=1) - 1
            new_indices = cumsum[~next_mask].reshape(B, -1)
            mask_indices = torch.cat((mask_indices, new_indices), dim=1)
            # add next glimpse to current mask
            next_mask = next_mask * current_mask
        return ~next_mask, mask_indices
