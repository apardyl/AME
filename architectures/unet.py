import torch
import torch.nn as nn

from torch.optim import Adam
from torchvision.transforms import CenterCrop

from architectures.utils import BaseArchitecture
from config import IMG_SIZE
from criterions.focal_loss import FocalLoss


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), enc_chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i + 1] + enc_chs[len(enc_chs) - i - 2], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(BaseArchitecture):
    def __init__(self, args, num_class=3, retain_dim=False, output_size=IMG_SIZE):
        super().__init__()
        self.encoder = Encoder(args.enc_channels)
        self.decoder = Decoder(args.dec_channels, args.enc_channels)
        self.head = nn.Conv2d(args.dec_channels[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.output_size = output_size

        self.criterion = torch.nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.lr_decay, last_epoch=-1)

    def forward(self, x):
        enc_ftrs = self.encoder(x["input"])
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = nn.functional.interpolate(out, self.output_size, mode="bilinear")
        return out

    @staticmethod
    def add_args(parser):
        parser.add_argument('--enc-channels',
                            help='encoder channels)',
                            type=int,
                            nargs="+",
                            default=(3, 16, 32, 64, 128, 256))
        parser.add_argument('--dec-channels',
                            help='decoder channels)',
                            type=int,
                            nargs="+",
                            default=(256, 128, 64, 32, 16))
        return parser

    def calculate_loss(self, out, batch):
        return self.criterion(out, batch["target"])

    def on_epoch_end(self):
        self.lr_scheduler.step()
