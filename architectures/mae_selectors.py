import torch
import torch.nn as nn

from torch.optim import Adam

from architectures.utils import BaseArchitecture
from config import IMG_SIZE, GLIMPSE_SIZE, GLIMPSES_W, GLIMPSES_H
from architectures.mae import mae_vit_large_patch16
from architectures.utils import WarmUpScheduler


class RandomMae(BaseArchitecture):
    def __init__(self, args):
        super().__init__()
        self.mae = mae_vit_large_patch16()
        checkpoint = torch.load("architectures/mae_vit_l_128x256.pth", map_location='cpu')["model"]
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']
        self.mae.load_state_dict(checkpoint, strict=False)
        # for p in self.mae.parameters():
        #     p.requires_grad = False
        self.num_glimpses = args.num_glimpses
        self.glimpse_selector = RandomGlimpseSelector()

        self.optimizer = Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = WarmUpScheduler(self.optimizer, args.num_samples // args.batch_size, args.lr_decay)

    def forward(self, x):
        mask = torch.full((x.shape[0], GLIMPSES_W * GLIMPSES_H), fill_value=False, device=x.device)
        mask_indices = torch.empty((x.shape[0], 0), dtype=torch.int64, device=x.device)
        losses = []
        for i in range(self.num_glimpses):
            mask, mask_indices = self.glimpse_selector(mask, mask_indices, i)
            latent = self.mae.forward_encoder(x, mask, mask_indices)
            pred = self.mae.forward_decoder(latent, mask, mask_indices)
            losses.append(self.mae.forward_loss(x, pred, mask))
        return {"out": pred, "mask": mask, "losses": losses}

    @staticmethod
    def add_args(parser):
        parser.add_argument('--encoder-embedding-dim',
                            help='encoder embedding size)',
                            type=int,
                            default=128)
        parser.add_argument('--dropout',
                            help='dropout)',
                            type=float,
                            default=0.0)
        return parser

    def calculate_loss(self, out, batch):
        """ Aggregate losses from all time steps"""
        loss = torch.mean(torch.stack(out["losses"]))
        return loss, {"MSE": float(loss)}

    def on_iter_end(self):
        self.lr_scheduler.step_iter()

    def on_epoch_end(self):
        self.lr_scheduler.step_epoch()


class CheckerboardMae(RandomMae):
    def __init__(self, args):
        super().__init__(args)
        self.glimpse_selector = CheckerboardGlimpseSelector()


class RandomGlimpseSelector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask, mask_indices, glimpse_num):
        """True in mask represents patches kept, False removed"""
        N, L = mask.shape
        new_glimpse_x = torch.randint(0, GLIMPSES_W - 2, size=(N, 1), device=mask.device)
        new_glimpse_y = torch.randint(0, GLIMPSES_H - 2, size=(N, 1), device=mask.device)
        glimpses = GLIMPSES_W * new_glimpse_y + new_glimpse_x
        glimpses = glimpses.repeat(1, 3)
        glimpses[:, 1] += 1
        glimpses[:, 2] += 2
        glimpses = torch.cat((glimpses, glimpses + GLIMPSES_W, glimpses + 2*GLIMPSES_W), dim=1)
        mask = mask.scatter(1, glimpses, torch.full_like(mask, fill_value=True))
        mask_indices = torch.cat((mask_indices, glimpses), dim=1)
        return mask, mask_indices


class CheckerboardGlimpseSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.coordinates = [(1, 1), (5, 1), (9, 1), (13, 1),
                            (1, 5), (5, 5), (9, 5), (13, 5)]

    def forward(self, mask, mask_indices, glimpse_num):
        N, L = mask.shape
        to_take = self.coordinates[glimpse_num]
        new_glimpse_x = torch.full((N, 1), fill_value=to_take[0], device=mask.device)
        new_glimpse_y = torch.full((N, 1), fill_value=to_take[1], device=mask.device)
        glimpses = GLIMPSES_W * new_glimpse_y + new_glimpse_x
        glimpses = glimpses.repeat(1, 3)
        glimpses[:, 1] += 1
        glimpses[:, 2] += 2
        glimpses = torch.cat((glimpses, glimpses + GLIMPSES_W, glimpses + 2*GLIMPSES_W), dim=1)
        mask.scatter_(1, glimpses, torch.full_like(mask, fill_value=True))
        mask_indices = torch.cat((mask_indices, glimpses), dim=1)

        return mask, mask_indices
