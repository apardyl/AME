import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.utils import IMAGENET_MEAN, IMAGENET_STD

_imagenet_mean = torch.tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(1)
_imagenet_std = torch.tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(1)

plt.rcParams.update({'figure.max_open_warning': 50})


def save_reconstructions(model, output, batch, vis_id, dump_path):
    mae = model.mae
    with torch.no_grad():
        B = batch[0].shape[0]
        steps = output['steps']
        reconstructed = torch.stack([x['out'] for x in steps], dim=1).reshape(
            B * len(steps), *output['out'].shape[1:])
        mask = torch.stack([x['mask'] for x in steps], dim=1)
        has_att_info = len(steps[1][2]) > 0
        if has_att_info:
            entropies = torch.stack([x['entropy'].cpu() for x in steps[1:]], dim=1)
            selection_maps = torch.stack([x['selection_map'].cpu() for x in steps[1:]], dim=1)

        reconstructed = torch.clip(
            (mae.unpatchify(reconstructed) * _imagenet_std + _imagenet_mean) * 255, 0, 255)
        reconstructed = reconstructed.reshape(B, len(steps), *reconstructed.shape[1:])

        target = torch.clip((batch[0].cpu() * _imagenet_std + _imagenet_mean) * 255, 0, 255)

        grid_h = mae.grid_size[0]
        grid_w = mae.grid_size[1]
        msk = mask.reshape(B, len(steps), grid_h, grid_w)

        # Dump visualizations
        os.makedirs(dump_path, exist_ok=True)
        for i in range(B):
            path = f"{dump_path}/{vis_id}_{i}.jpg"
            fig, axs = plt.subplots(5 if has_att_info else 3, 9)
            for s in range(len(steps)):
                fig.set_size_inches(20, 7 if has_att_info else 5)
                fig.tight_layout()

                trg_image = np.array(target[i].permute(1, 2, 0), dtype=np.uint8)
                axs[0, s].imshow(trg_image)
                axs[0, s].set_title('input')

                msk_image = np.array(msk[i][s])
                axs[1, s].imshow(msk_image)
                axs[1, s].set_title('mask')

                out_image = np.array(reconstructed[i][s].permute(1, 2, 0), dtype=np.uint8)
                axs[2, s].imshow(out_image)
                axs[2, s].set_title('out')

                if has_att_info:
                    if s < entropies.shape[1]:
                        axs[3, s].imshow(entropies[i][s][0])
                        axs[3, s].set_title('entropy')

                        axs[4, s].imshow(selection_maps[i][s][0])
                        axs[4, s].set_title('next glimpse')
                    else:
                        axs[3, s].axis('off')
                        axs[4, s].axis('off')

            fig.savefig(path)
            plt.cla()
