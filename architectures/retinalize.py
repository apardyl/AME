import torch
import torch.nn.functional as F


class Retinalizer:
    # slow and inefficient, but good enough for one experiment
    # untested for glimpse_size != 3
    def __init__(self, model, args):
        self.num_glimpses = model.num_glimpses
        self.glimpse_size = args.glimpse_size
        self.patch_size = model.mae.patch_embed.patch_size[0]
        self.glimpse_size_px = self.glimpse_size * self.patch_size
        self.grid_h = model.mae.grid_size[0]
        self.grid_w = model.mae.grid_size[1]

    def __call__(self, images, glimpses):
        images2 = torch.zeros_like(images)
        for b in range(images.shape[0]):
            for g in glimpses:
                x = int(g[b][0])
                h = x // self.grid_w
                w = x % self.grid_w
                h_px = h * self.patch_size
                w_px = w * self.patch_size
                images2[b, :, h_px:h_px + self.glimpse_size_px, w_px:w_px + self.glimpse_size_px] = \
                    self.get_retinal_glimpse(images[b], h_px, w_px, glimpse_size_px=self.glimpse_size_px)
        return images2

    @staticmethod
    def get_retinal_glimpse(image, top, left, glimpse_size_px=48):
        # adapted from https://github.com/soroushseifi/glimpse-attend-explore/blob/main/utils.py
        device = image.device
        retina_size = glimpse_size_px // 3
        mask_s1 = torch.ones([retina_size, retina_size], device=device)
        channels = image.shape[-3]

        pad_size = (retina_size // 2, retina_size // 2, retina_size // 2, retina_size // 2)
        offsets_right = [2 * glimpse_size_px // 3, (5 / 6) * glimpse_size_px, glimpse_size_px]
        offsets_left = [glimpse_size_px // 3, glimpse_size_px // 6, 0]

        borders = [top + offsets_left[0], top + offsets_right[0],
                   left + offsets_left[0], left + offsets_right[0]]
        scale1 = image[..., borders[0]:borders[1], borders[2]:borders[3]]
        # scale 2 - x2 downscaled mid-part of the glimpse
        borders = [top + offsets_left[1], top + offsets_right[1],
                   left + offsets_left[1], left + offsets_right[1]]
        scale2_t = image[..., int(borders[0]):int(borders[1]), int(borders[2]):int(borders[3])]
        # x2 downscale
        scale2_ds = F.interpolate(torch.unsqueeze(scale2_t, 0), size=[retina_size, retina_size], mode='bilinear',
                                  align_corners=False)
        scale2 = torch.squeeze(
            F.interpolate(scale2_ds, size=[2 * retina_size, 2 * retina_size], mode='bilinear', align_corners=False))
        # scale 3 - x3 downscaled outer part of the glimpse
        borders = [top + offsets_left[2], top + offsets_right[2],
                   left + offsets_left[2], left + offsets_right[2]]
        scale3_t = image[..., borders[0]:borders[1], borders[2]:borders[3]]
        # x3 downscale
        scale3_ds = F.interpolate(torch.unsqueeze(scale3_t, 0), size=[retina_size, retina_size], mode='bilinear',
                                  align_corners=False)
        scale3 = torch.squeeze(
            F.interpolate(scale3_ds, size=[glimpse_size_px, glimpse_size_px], mode='bilinear', align_corners=False))
        # pad scale1 with zeros so that it has the same size as scale2
        scale1_padded = F.pad(scale1, pad_size, "constant", 0)
        mask_s12 = F.pad(mask_s1, pad_size, "constant", 0)
        # fill the surrondings of scale 1 with scale2 (scale1+scale2)
        scale12 = (1 - mask_s12) * scale2 + mask_s12 * scale1_padded
        # pad (scale1+scale2) to have the same size as scale 3
        scale12_padded = F.pad(scale12, pad_size, "constant", 0)
        mask_s123 = F.pad(mask_s12, pad_size, "constant", 0)
        # fill the surronding of (scale1+scale2) with scale 3 (scale2+scale1+scale3=glimpse)
        scale123 = (1 - mask_s123) * scale3 + mask_s123 * scale12_padded
        glimpse = scale123.view(channels, glimpse_size_px, glimpse_size_px)
        return glimpse
