import argparse
import os
import time
from pathlib import Path

import torch

import architectures


def dict_to_device(d: dict, device):
    for key in d.keys():
        if torch.is_tensor(d[key]):
            d[key] = d[key].to(device, non_blocking=True)


def save_model(model, path, is_best):
    Path(path).mkdir(parents=True, exist_ok=True)
    if is_best:
        torch.save(model.state_dict(), os.path.join(path, "best.pth"))
    torch.save(model.state_dict(), os.path.join(path, "last.pth"))


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def create_checkpoint_dir(base_checkpoint_dir):
    """
    Creates an arbitrary checkpoint dir. Either with a provided name or just with a date, i.e.
    checkpoints/test_ --> checkpoints/test_<date>
    checkpoints/ --> checkpoints/<date>
    Returns: checkpoint dir path

    """
    current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
    checkpoint_dir = base_checkpoint_dir + current_time
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def parse_args():
    """
    Parse arguments and modify them if needed
    Returns: args (Namespace)
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint-dir',
                        help='directory where models will be saved and loaded from',
                        type=str,
                        default='checkpoints/')
    parser.add_argument('--load-model-path',
                        help='load model from pth',
                        type=str,
                        default=None)
    parser.add_argument('--batch-size',
                        help='mini batch size',
                        type=int,
                        default=64)
    parser.add_argument('--num-glimpses',
                        help='number of glimpses to take',
                        type=int,
                        default=8)
    parser.add_argument('--num-workers',
                        help='data loader workers',
                        type=int,
                        default=0)
    parser.add_argument('--epochs',
                        help='number of epochs',
                        type=int,
                        default=50)
    parser.add_argument('--num-samples',
                        help='number of images to sample in each training epoch',
                        type=int,
                        default=20000)
    parser.add_argument('--clip',
                        help='gradient clipping',
                        type=float,
                        default=1.0)
    parser.add_argument('--lr',
                        help='learning_rate',
                        type=float,
                        default=1e-3)
    parser.add_argument('--lr-decay',
                        help='learning rate decay each epoch',
                        type=float,
                        default=0.97)
    parser.add_argument('--weight-decay',
                        help='weight_decay',
                        type=float,
                        default=0)
    parser.add_argument('--threshold',
                        help='threshold above which pixel is classified as positive',
                        type=float,
                        default=0.5)
    parser.add_argument('--alpha',
                        help='alpha class weighting for focal loss)',
                        type=float,
                        default=0.9)
    parser.add_argument('--gamma',
                        help='gamma for focal loss)',
                        type=float,
                        default=2.0)
    parser.add_argument('--arch',
                        help='architecture name',
                        type=str,
                        default="RandomMae")
    parser.add_argument('--test',
                        help='test model',
                        action='store_true',
                        default=False)
    parser.add_argument('--max-samples',
                        help='number of samples to take from each HDF',
                        type=int,
                        default=100000)
    parser.add_argument('--use-fp16',
                        help='sets models precision to FP16. Default is FP32',
                        action='store_true',
                        default=False)
    parser.add_argument('--patience',
                        help='early stop training if valid performance doesn’t improve for N consecutive validation runs',
                        type=int,
                        default=10)

    args = parser.parse_args()

    arch = getattr(architectures, args.arch)
    if arch is None:
        raise RuntimeError(f"Wrong architecture provided. Possible architectures: {archs.keys()}")

    args.arch = arch

    return args
