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
        self.mae.load_state_dict(torch.load("architectures/mae_visualize_vit_large.pth", map_location='cpu')["model"], strict=True)
        for p in self.mae.parameters():
            p.requires_grad = False
        self.glimpse_selector = RandomGlimpseSelector()

        self.criterion = torch.nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.lr_decay, last_epoch=-1)

    def forward(self, x):
        with torch.no_grad():
            mask = torch.full((x.shape[0], GLIMPSES_W * GLIMPSES_H), fill_value=False, device=x.device)
            for i in range(8):
                mask = self.glimpse_selector(mask)
                latent, mask = self.mae.forward_encoder(x, mask)
                pred = self.mae.forward_decoder(latent, mask)
                pred = self.mae.unpatchify(pred)
        return {"out": pred, "mask": mask}

    def next_glimpse(self, *args):
        return self.glimpse_selector(args)

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


class RandomGlimpseSelector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask):
        N, L = mask.shape
        new_glimpse_x = torch.randint(0, GLIMPSES_W // 2 - 1, size=(N, 1), device=mask.device)
        new_glimpse_y = torch.randint(0, GLIMPSES_H // 2 - 1, size=(N, 1), device=mask.device)
        glimpses = 2 * GLIMPSES_H * new_glimpse_x + 2 * new_glimpse_y
        glimpses_down = glimpses + 1
        glimpses_right = glimpses + GLIMPSES_H
        glimpses_down_right = glimpses_right + 1

        glimpses = torch.cat((glimpses, glimpses_down, glimpses_right, glimpses_down_right), dim=1)
        mask.scatter_(1, glimpses, torch.full_like(mask, fill_value=True))

        return mask
