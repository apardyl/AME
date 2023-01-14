import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.utils import IMAGENET_MEAN, IMAGENET_STD

_imagenet_mean = torch.tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(1)
_imagenet_std = torch.tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(1)

plt.rcParams.update({'figure.max_open_warning': 50})


def upscale_patch_values(mask, patch_size=(16, 16)):
    return mask.repeat(patch_size[1], axis=1).repeat(patch_size[0], axis=0)


def save_reconstructions(model, output, batch, vis_id, dump_path):
    mae = model.mae
    with torch.no_grad():
        B = batch[0].shape[0]
        steps = output['steps']
        reconstructed = torch.stack([x['out'] for x in steps], dim=1).reshape(
            B * len(steps), *output['out'].shape[1:])
        mask = torch.stack([x['mask'] for x in steps], dim=1)
        has_att_info = len(steps[3]['entropy']) > 0

        reconstructed = torch.clip(
            (mae.unpatchify(reconstructed) * _imagenet_std + _imagenet_mean) * 255, 0, 255)
        reconstructed = reconstructed.reshape(B, len(steps), *reconstructed.shape[1:])

        target = torch.clip((batch[0].cpu() * _imagenet_std + _imagenet_mean) * 255, 0, 255)

        grid_h = mae.grid_size[0]
        grid_w = mae.grid_size[1]
        msk = mask.reshape(B, len(steps), grid_h, grid_w)
        patch_size = mae.patch_embed.patch_size

        # Dump visualizations
        os.makedirs(dump_path, exist_ok=True)
        for i in range(B):
            path = os.path.join(dump_path, f"{vis_id}_{i}.jpg")
            sub_path = os.path.join(dump_path, f'{vis_id}_{i}')
            os.makedirs(sub_path, exist_ok=True)
            fig, axs = plt.subplots(5 if has_att_info else 3, len(steps))
            fig.set_size_inches(len(steps) * 2, 7 if has_att_info else 5)
            fig.tight_layout()
            for s in range(len(steps)):
                trg_image = np.array(target[i].permute(1, 2, 0), dtype=np.uint8)
                msk_image = np.array(msk[i][s])
                out_image = np.array(reconstructed[i][s].permute(1, 2, 0), dtype=np.uint8)

                axs[0, s].imshow(trg_image)
                axs[0, s].set_title('input')

                axs[1, s].imshow(msk_image)
                axs[1, s].set_title('mask')

                axs[2, s].imshow(out_image)
                axs[2, s].set_title('out')

                msk_upscales = upscale_patch_values(msk_image, patch_size)
                plt.imsave(os.path.join(sub_path, f'{s}_input.jpg'), trg_image)
                plt.imsave(os.path.join(sub_path, f'{s}_mask.jpg'), msk_upscales)
                plt.imsave(os.path.join(sub_path, f'{s}_masked.jpg'),
                           upscale_patch_values(trg_image * msk_upscales[..., np.newaxis], patch_size))
                plt.imsave(os.path.join(sub_path, f'{s}_out.jpg'), out_image)

                if has_att_info:
                    if s > 0 and 'entropy' in steps[s - 1]:
                        axs[3, s].imshow(steps[s - 1]['entropy'][i][0].cpu())
                        axs[3, s].set_title('entropy')

                        axs[4, s].imshow(steps[s - 1]['selection_map'][i][0].cpu())
                        axs[4, s].set_title('next glimpse')

                        plt.imsave(os.path.join(sub_path, f'{s}_entropy.jpg'),
                                   upscale_patch_values(steps[s - 1]['entropy'][i][0].cpu().numpy(), patch_size))
                        plt.imsave(os.path.join(sub_path, f'{s}_next.jpg'),
                                   upscale_patch_values(steps[s - 1]['selection_map'][i][0].cpu().numpy(), patch_size))
                    else:
                        axs[3, s].axis('off')
                        axs[4, s].axis('off')

            fig.savefig(path)
            plt.cla()
