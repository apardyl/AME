import numpy as np
import os
import torch

from PIL import Image


class ReconstructionMetricsLogger:
    def __init__(self, epoch: int, mode: str, args):
        self.epoch = epoch
        self.mode = mode  # train, valid or test
        self.checkpoint_dir = args.checkpoint_dir
        self.total_samples = []
        self.losses = []
        self.rmses = []
        self.tina = []

        self.visualizations_to_save = 5  # For valid and test modes create this number of visualizations

    def log(self, output, batch: dict, loss: dict):
        bsz = output.shape[0]
        self.losses.append(bsz * float(loss["MSE"]))
        rmse = torch.mean(torch.sqrt(torch.mean((output - batch["target"])**2, [1, 2, 3])))
        self.rmses.append(bsz * float(rmse))
        tinas_stupid_metric = torch.mean(torch.sqrt(torch.sum((output - batch["target"]) ** 2, 1)), [0, 1, 2])
        self.tina.append(bsz * float(tinas_stupid_metric))

        # Dump visualizations
        if self.visualizations_to_save > 0 and (self.mode == "valid" or self.mode == "test"):
            dump_path = f"{self.checkpoint_dir}/{self.epoch}"
            os.makedirs(dump_path, exist_ok=True)
            output = torch.clip(output, 0, 1).detach().cpu() * 255
            target = batch["target"].cpu() * 255
            for i in range(bsz):
                out_image = np.array(output[i].permute(1, 2, 0), dtype=np.uint8)
                out_image = Image.fromarray(out_image)
                out_image.save(f"{dump_path}/{self.visualizations_to_save}_output.png")
                trg_image = np.array(target[i].permute(1, 2, 0), dtype=np.uint8)
                trg_image = Image.fromarray(trg_image)
                trg_image.save(f"{dump_path}/{self.visualizations_to_save}_target.png")

                self.visualizations_to_save -= 1
                if self.visualizations_to_save <= 0:
                    break

    def get_epoch_result(self, tb_writer, eps=1e-8):
        loss = sum(self.losses) / (len(self.losses) + eps)
        rmse = sum(self.rmses) / (len(self.rmses) + eps)
        tina = sum(self.tina) / (len(self.tina) + eps)

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
        return f"{prefix}Loss: {10**2*r['loss']:.3f} RMSE: {r['RMSE']} Tina: {r['Tina']}"
