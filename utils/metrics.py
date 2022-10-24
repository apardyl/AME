import torch


class ReconstructionMetricsLogger:
    def __init__(self, mode: str, args):
        self.mode = mode  # train, valid or test
        self.total_samples = []
        self.losses = []
        self.rmses = []

    def log(self, output, batch: dict, loss: dict):
        bsz = output.shape[0]
        self.losses.append(bsz * loss["MSE"])
        rmse = torch.mean(torch.sqrt(torch.mean((output - batch["target"])**2, [1, 2, 3])))
        self.rmses.append(bsz * rmse)

    def get_epoch_result(self, tb_writer, epoch, eps=1e-8):
        loss = sum(self.losses) / (len(self.losses) + eps)
        rmse = sum(self.rmses) / (len(self.rmses) + eps)

        tb_writer.add_scalar(f'Loss/{self.mode}', loss, epoch)
        tb_writer.add_scalar(f'RMSE/{self.mode}', rmse, epoch)

        return {
            "loss": loss,
            "RMSE": rmse,
        }

    @staticmethod
    def result_to_string(result: dict, prefix=""):
        r = result
        return f"{prefix}Loss: {10**2*r['loss']:.3f} RMSE: {r['RMSE']}"
