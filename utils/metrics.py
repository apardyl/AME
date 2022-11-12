import numpy as np
import os
import torch

from PIL import Image

from config import GLIMPSES_H, GLIMPSES_W, GLIMPSE_SIZE
from data_utils.datasets import IMAGENET_MEAN, IMAGENET_STD


class ReconstructionMetricsLogger:
    def __init__(self, epoch: int, mode: str, args):
        self.epoch = epoch
        self.mode = mode  # train, valid or test
        self.checkpoint_dir = args.checkpoint_dir
        self.total_samples = 0
        self.losses = []
        self.rmses = []
        self.tina = []

        self.visualizations_to_save = 10  # For valid and test modes create this number of visualizations

    def log(self, output, batch: dict, loss: dict):
        with torch.no_grad():
            bsz = output["out"].shape[0]
            self.total_samples += bsz
            self.losses.append(bsz * float(loss["MSE"]))
            reconstructed = torch.clip((output["out"].detach().cpu() * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255)
            target = torch.clip((batch["target"].cpu() * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255)

            # Paste ground truth glimpses into reconstructed image
            reconstructed_pasted = reconstructed.clone()
            a = patchify(reconstructed_pasted)
            a[output["mask"], :] = patchify(target)[output["mask"], :]
            reconstructed_pasted = unpatchify(a)

            rmse = torch.mean(torch.sqrt(torch.mean((reconstructed - target)**2, [1, 2, 3])))
            self.rmses.append(bsz * float(rmse))
            tinas_stupid_metric = torch.mean(torch.sqrt(torch.sum((reconstructed - target) ** 2, 1)), [0, 1, 2])
            self.tina.append(bsz * float(tinas_stupid_metric))

            # Dump visualizations
            if self.visualizations_to_save > 0 and (self.mode == "valid" or self.mode == "test"):
                dump_path = f"{self.checkpoint_dir}/{self.epoch}"
                os.makedirs(dump_path, exist_ok=True)
                for i in range(bsz):
                    out_image = np.array(reconstructed[i].permute(1, 2, 0), dtype=np.uint8)
                    out_image = Image.fromarray(out_image)
                    out_image.save(f"{dump_path}/{self.visualizations_to_save}_reconstructed.png")
                    out_image = np.array(reconstructed_pasted[i].permute(1, 2, 0), dtype=np.uint8)
                    out_image = Image.fromarray(out_image)
                    out_image.save(f"{dump_path}/{self.visualizations_to_save}_reconstructed_pasted.png")
                    trg_image = np.array(target[i].permute(1, 2, 0), dtype=np.uint8)
                    trg_image = Image.fromarray(trg_image)
                    trg_image.save(f"{dump_path}/{self.visualizations_to_save}_target.png")

                    self.visualizations_to_save -= 1
                    if self.visualizations_to_save <= 0:
                        break

    def get_epoch_result(self, tb_writer):
        loss = sum(self.losses) / self.total_samples
        rmse = sum(self.rmses) / self.total_samples
        tina = sum(self.tina) / self.total_samples

        tb_writer.add_scalar(f'Loss/{self.mode}', loss, self.epoch)
        tb_writer.add_scalar(f'RMSE/{self.mode}', rmse, self.epoch)
        tb_writer.add_scalar(f'Tina/{self.mode}', tina, self.epoch)

        return {
            "loss": loss,
            "RMSE": rmse,
            "Tina": tina,
        }

    @staticmethod
    def result_to_string(result: dict, prefix=""):
        r = result
        return f"{prefix}Loss: {10**2*r['loss']:.3f} RMSE: {r['RMSE']:.2f} Tina: {r['Tina']:.2f}"


def patchify(imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """

    p = GLIMPSE_SIZE
    x = imgs.reshape(shape=(imgs.shape[0], 3, GLIMPSES_H, p, GLIMPSES_W, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], GLIMPSES_H * GLIMPSES_W, p * p * 3))
    return x


def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = GLIMPSE_SIZE
    x = x.reshape(shape=(x.shape[0], GLIMPSES_H, GLIMPSES_W, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, p * GLIMPSES_H, p * GLIMPSES_W))
    return imgs