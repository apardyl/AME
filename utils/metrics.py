import torch


class ReconstructionMetricsLogger:
    def __init__(self, args):
        self.total_samples = []
        self.losses = []
        self.rmses = []

    def log(self, output, target, loss):
        bsz = output.shape[0]
        self.losses.append(bsz * loss)
        self.rmses.append(0)  # TODO, Tina does it wrong imo cause she averages it over batch

    def get_epoch_result(self, eps=1e-8):
        loss = sum(self.losses) / (len(self.losses) + eps)
        rmse = sum(self.rmses) / (len(self.rmses) + eps)

        return {
            "loss": loss,
            "RMSE": rmse,
        }

    @staticmethod
    def result_to_string(result: dict, prefix=""):
        r = result
        return f"{prefix}Loss: {10**2*r['loss']:.3f} RMSE: {r['RMSE']}"
