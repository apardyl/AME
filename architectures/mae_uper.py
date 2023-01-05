# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

from architectures.mae_utils import get_2d_sincos_pos_embed, Block
from architectures.mae import MaskedAutoencoderViT
from mmseg.models.necks.featurepyramid import Feature2Pyramid
from mmseg.models.decode_heads.uper_head import UPerHead


class MaskedAutoencoderViTUPer(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, out_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, out_chans, embed_dim,
                         depth, num_heads, decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_pix_loss)

        self.neck = Feature2Pyramid(768, rescales=[4, 2, 1, 0.5], norm_cfg=dict(type='BN', requires_grad=True))  # original on 768
        self.head = UPerHead(pool_scales=(1, 2, 3, 6),
                             in_channels=[768, 768, 768, 768],
                             in_index=[0, 1, 2, 3],
                             channels=768,
                             dropout_ratio=0.1,
                             num_classes=out_chans
                             )

    def forward_decoder(self, x, mask, patch_indices):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        x_ = self.mask_token.repeat(*mask.shape, 1)
        x_.scatter_(1, patch_indices.unsqueeze(2).repeat(1, 1, x_.shape[2]), x[:, 1:, :])
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        outs = []
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)
            if i in [1, 3, 5, 7]:
                out = x[:, 1:]
                B, _, C = out.shape
                out = out.reshape(B, self.grid_size[0], self.grid_size[1],C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        outs = tuple(outs)
        outs = self.neck(outs)
        out = self.head(outs)
        out = nn.functional.interpolate(out, None, scale_factor=4, mode='bilinear', align_corners=False)
        return out

    @torch.no_grad()
    def segmentation_output(self, pred):
        return torch.argmax(pred, dim=1)


def mae_vit_large_patch16_dec512d8b_uper(**kwargs):
    model = MaskedAutoencoderViTUPer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=768, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b_uper
