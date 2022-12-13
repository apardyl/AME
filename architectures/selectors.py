import abc

import torch
import torch.nn as nn

from architectures.glimpse_mae import BaseGlimpseMae


class BaseGlimpseSelector:
    def __init__(self, model: BaseGlimpseMae):
        super().__init__()

        self.model = model
        self.glimpse_size = model.glimpse_size
        self.grid_h = model.mae.grid_size[0]
        self.grid_w = model.mae.grid_size[1]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, current_mask, mask_indices, glimpse_num):
        raise NotImplemented()


class RandomGlimpseSelector(BaseGlimpseSelector):
    def __init__(self, model):
        super().__init__(model)
        assert self.glimpse_size == 3  # other sizes not supported yet

    def forward(self, mask, mask_indices, glimpse_num):
        """True in mask represents patches kept, False removed"""
        N, L = mask.shape
        new_glimpse_x = torch.randint(0, self.grid_w - 2, size=(N, 1), device=mask.device)
        new_glimpse_y = torch.randint(0, self.grid_h - 2, size=(N, 1), device=mask.device)
        glimpses = self.grid_w * new_glimpse_y + new_glimpse_x
        glimpses = glimpses.repeat(1, 3)
        glimpses[:, 1] += 1
        glimpses[:, 2] += 2
        glimpses = torch.cat((glimpses, glimpses + self.grid_w, glimpses + 2 * self.grid_w), dim=1)
        mask = mask.scatter(1, glimpses, torch.full_like(mask, fill_value=True))
        mask_indices = torch.cat((mask_indices, glimpses), dim=1)
        return mask, mask_indices


class CheckerboardGlimpseSelector(BaseGlimpseSelector):
    def __init__(self, model):
        super().__init__(model)
        self.coordinates = [(1, 1), (5, 1), (9, 1), (13, 1),
                            (1, 5), (5, 5), (9, 5), (13, 5)]
        assert self.glimpse_size == 3  # other sizes not supported yet

    def forward(self, mask, mask_indices, glimpse_num):
        N, L = mask.shape
        to_take = self.coordinates[glimpse_num]
        new_glimpse_x = torch.full((N, 1), fill_value=to_take[0], device=mask.device)
        new_glimpse_y = torch.full((N, 1), fill_value=to_take[1], device=mask.device)
        glimpses = self.grid_w * new_glimpse_y + new_glimpse_x
        glimpses = glimpses.repeat(1, 3)
        glimpses[:, 1] += 1
        glimpses[:, 2] += 2
        glimpses = torch.cat((glimpses, glimpses + self.grid_w, glimpses + 2 * self.grid_w), dim=1)
        mask.scatter_(1, glimpses, torch.full_like(mask, fill_value=True))
        mask_indices = torch.cat((mask_indices, glimpses), dim=1)

        return mask, mask_indices


class AttentionGlimpseSelector(BaseGlimpseSelector):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, current_mask, mask_indices, glimpse_num):
        with torch.no_grad():
            current_mask = ~current_mask
            # get self attention weights
            attn = self.model.mae.last_attn[7][..., 1:, 1:]
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
