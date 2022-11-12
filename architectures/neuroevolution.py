import torch
import torch.nn as nn

from torch.optim import Adam

from architectures.utils import BaseArchitecture
from config import IMG_SIZE, GLIMPSE_SIZE, GLIMPSES_W, GLIMPSES_H
from architectures.mae import mae_vit_large_patch16


class RandomMae(BaseArchitecture):
    def __init__(self, args):
        super().__init__()
        self.mae = mae_vit_large_patch16()
        checkpoint = torch.load("architectures/mae_vit_l_128x256.pth", map_location='cpu')["model"]
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']
        self.mae.load_state_dict(checkpoint, strict=False)
        for p in self.mae.parameters():
            p.requires_grad = False
        self.num_glimpses = args.num_glimpses
        self.glimpse_selector = RandomGlimpseSelector()

        self.criterion = torch.nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.lr_decay, last_epoch=-1)

    def forward(self, x):
        with torch.no_grad():
            mask = torch.full((x.shape[0], GLIMPSES_W * GLIMPSES_H), fill_value=False, device=x.device)
            for i in range(self.num_glimpses):
                mask = self.glimpse_selector(mask, i)
                latent = self.mae.forward_encoder(x, mask)
                pred = self.mae.forward_decoder(latent, mask)
                pred = self.mae.unpatchify(pred)
        return {"out": pred, "mask": mask}

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
        loss = self.criterion(out["out"], batch["target"])
        return loss, {"MSE": float(loss)}

    def on_epoch_end(self):
        self.lr_scheduler.step()


class CheckerboardMae(RandomMae):
    def __init__(self, args):
        super().__init__(args)
        self.glimpse_selector = CheckerboardGlimpseSelector()


class RandomGlimpseSelector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask, glimpse_num):  # does not work for batch size > 1 because it can overlap
        N, L = mask.shape
        new_glimpse_x = torch.randint(0, GLIMPSES_W // 2 - 1, size=(N, 1), device=mask.device)
        new_glimpse_y = torch.randint(0, GLIMPSES_H // 2 - 1, size=(N, 1), device=mask.device)
        glimpses = 2 * GLIMPSES_W * new_glimpse_y + 2 * new_glimpse_x
        glimpses_down = glimpses + 1
        glimpses_right = glimpses + GLIMPSES_W
        glimpses_down_right = glimpses_right + 1

        glimpses = torch.cat((glimpses, glimpses_down, glimpses_right, glimpses_down_right), dim=1)
        mask.scatter_(1, glimpses, torch.full_like(mask, fill_value=True))

        return mask


class CheckerboardGlimpseSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.coordinates = [(1, 1), (5, 1), (9, 1), (13, 1),
                            (1, 5), (5, 5), (9, 5), (13, 5)]

    def forward(self, mask, glimpse_num):
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

        return mask
