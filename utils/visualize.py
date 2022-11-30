import os

import numpy as np
import torch
from PIL import Image

from data_utils.datasets import IMAGENET_MEAN, IMAGENET_STD

_imagenet_mean = torch.tensor(IMAGENET_MEAN).unsqueeze(1).unsqueeze(1)
_imagenet_std = torch.tensor(IMAGENET_STD).unsqueeze(1).unsqueeze(1)


def save_reconstructions(mae, output, batch, visualizations_to_save, dump_path):
    with torch.no_grad():
        bsz = output["out"].shape[0]
        reconstructed = torch.clip(
            (mae.unpatchify(output["out"]).detach().cpu() * _imagenet_std + _imagenet_mean) * 255, 0, 255)
        target = torch.clip((batch[0].cpu() * _imagenet_std + _imagenet_mean) * 255, 0, 255)

        # Paste ground truth glimpses into reconstructed image
        reconstructed_pasted = reconstructed.clone()
        a = mae.patchify(reconstructed_pasted)
        a[output["mask"], :] = mae.patchify(target)[output["mask"].cpu(), :]
        reconstructed_pasted = mae.unpatchify(a)

        # Dump visualizations
        if visualizations_to_save > 0:
            os.makedirs(dump_path, exist_ok=True)
            for i in range(bsz):
                out_image = np.array(reconstructed[i].permute(1, 2, 0), dtype=np.uint8)
                out_image = Image.fromarray(out_image)
                out_image.save(f"{dump_path}/{visualizations_to_save}_reconstructed.png")
                out_image = np.array(reconstructed_pasted[i].permute(1, 2, 0), dtype=np.uint8)
                out_image = Image.fromarray(out_image)
                out_image.save(f"{dump_path}/{visualizations_to_save}_reconstructed_pasted.png")
                trg_image = np.array(target[i].permute(1, 2, 0), dtype=np.uint8)
                trg_image = Image.fromarray(trg_image)
                trg_image.save(f"{dump_path}/{visualizations_to_save}_target.png")

                visualizations_to_save -= 1
                if visualizations_to_save <= 0:
                    break
