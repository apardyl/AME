import torch
import torch.nn as nn

from torch.optim import Adam

from architectures.utils import BaseArchitecture
from config import IMG_SIZE, GLIMPSE_SIZE
from architectures.mae import mae_vit_large_patch16


class RandomMae(BaseArchitecture):
    def __init__(self, args):
        super().__init__()
        self.mae = mae_vit_large_patch16()
        self.mae.load_state_dict(torch.load("architectures/mae_visualize_vit_large.pth", map_location='cpu')["model"], strict=True)
        for p in self.mae.parameters():
            p.requires_grad = False

        self.criterion = torch.nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.lr_decay, last_epoch=-1)

    def forward(self, x):
        with torch.no_grad():
            latent, mask, ids_restore = self.mae.forward_encoder(x, 0.75)
            pred = self.mae.forward_decoder(latent, ids_restore)
            pred = self.mae.unpatchify(pred)
        return pred

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
        loss = self.criterion(out, batch["target"])
        return loss, {"MSE": float(loss)}

    def on_epoch_end(self):
        self.lr_scheduler.step()
