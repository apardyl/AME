import torch

from torch.optim import Adam

from architectures.selectors import RandomGlimpseSelector, CheckerboardGlimpseSelector, AttentionGlimpseSelector
from architectures.utils import BaseArchitecture
from config import GLIMPSES_W, GLIMPSES_H, IMG_SIZE
from architectures.mae import mae_vit_large_patch16
from architectures.utils import WarmUpScheduler


class RandomMae(BaseArchitecture):
    def __init__(self, args):
        super().__init__()
        self.mae = mae_vit_large_patch16(img_size=(IMG_SIZE[1], IMG_SIZE[0]))
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
            latent = self.mae.forward_encoder(x, mask_indices)
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


class AttentionMae(RandomMae):
    def __init__(self, args):
        super().__init__(args)
        self.glimpse_selector = AttentionGlimpseSelector()

    def forward(self, x):
        mask = torch.full((x.shape[0], GLIMPSES_W * GLIMPSES_H), fill_value=False, device=x.device)
        mask_indices = torch.empty((x.shape[0], 0), dtype=torch.int64, device=x.device)
        losses = []
        with torch.no_grad():
            latent = self.mae.forward_encoder(x, mask_indices)
            pred = self.mae.forward_decoder(latent, mask, mask_indices)
        for i in range(self.num_glimpses):
            mask, mask_indices = self.glimpse_selector(self.mae, mask, mask_indices, i)
            latent = self.mae.forward_encoder(x, mask_indices)
            pred = self.mae.forward_decoder(latent, mask, mask_indices)
            losses.append(self.mae.forward_loss(x, pred, mask))
        return {"out": pred, "mask": mask, "losses": losses}
