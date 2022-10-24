import torch


class ReconstructionMetricsLogger:
    def __init__(self, mode: str, args):
        self.mode = mode  # train, valid or test
        self.total_samples = []
        self.losses = []
        self.rmses = []
        self.tina = []

    def log(self, output, batch: dict, loss: dict):
        bsz = output.shape[0]
        self.losses.append(bsz * float(loss["MSE"]))
        rmse = torch.mean(torch.sqrt(torch.mean((output - batch["target"])**2, [1, 2, 3])))
        self.rmses.append(bsz * float(rmse))
        tinas_stupid_metric = torch.mean(torch.sqrt(torch.sum((output - batch["target"]) ** 2, 1)), [0, 1, 2])
        self.tina.append(bsz * float(tinas_stupid_metric))

    def get_epoch_result(self, tb_writer, epoch, eps=1e-8):
        loss = sum(self.losses) / (len(self.losses) + eps)
        rmse = sum(self.rmses) / (len(self.rmses) + eps)
        tina = sum(self.tina) / (len(self.tina) + eps)

        tb_writer.add_scalar(f'Loss/{self.mode}', loss, epoch)
        tb_writer.add_scalar(f'RMSE/{self.mode}', rmse, epoch)
        tb_writer.add_scalar(f'Tina/{self.mode}', tina, epoch)

        return {
            "loss": loss,
            "RMSE": rmse,
            "Tina": tina,
        }

    @staticmethod
    def result_to_string(result: dict, prefix=""):
        r = result
        return f"{prefix}Loss: {10**2*r['loss']:.3f} RMSE: {r['RMSE']} Tina: {r['Tina']}"
