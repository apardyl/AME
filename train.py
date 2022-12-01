import argparse
import inspect
import logging
import random
import sys

import pytorch_lightning
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin
from torch.cuda.amp import GradScaler

import architectures
import datasets
import wandb

random.seed(1)
torch.manual_seed(1)


def main():
    parser = argparse.ArgumentParser(
        prog='TrainWhereToLookNext'
    )

    archs = {k: v for k, v in inspect.getmembers(architectures, inspect.isclass) if
             issubclass(v, pytorch_lightning.LightningModule) and not inspect.isabstract(v)}
    dss = {k: v for k, v in inspect.getmembers(datasets, inspect.isclass) if
           issubclass(v, pytorch_lightning.LightningDataModule) and not inspect.isabstract(
               v) and k != 'LightningDataModule'}

    parser.add_argument('dataset', help='Dataset to use', choices=dss.keys())
    parser.add_argument('arch', help='Model architecture', choices=archs.keys())
    main_args = parser.parse_args(sys.argv[1:3])

    arch_class = archs[main_args.arch]
    data_module_class = dss[main_args.dataset]

    parser = argparse.ArgumentParser(
        prog=f'TrainWhereToLookNext {main_args.dataset} {main_args.arch}',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser = arch_class.add_argparse_args(parser)
    parser = data_module_class.add_argparse_args(parser)
    parser.add_argument('--epochs',
                        help='number of epochs',
                        type=int,
                        default=50)
    parser.add_argument('--use-fp16',
                        help='sets models precision to FP16. Default is FP32',
                        action='store_true',
                        default=False)
    parser.add_argument('--load-model-path',
                        help='load model from pth',
                        type=str,
                        default=None)

    args = parser.parse_args(sys.argv[3:])

    data_module = data_module_class(args)
    model = arch_class(args)

    model.load_pretrained_mae()
    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))

    logging.info(model)
    logging.info(
        f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters\n')
    logging.info(
        f'The model has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} frozen parameters\n')

    plugins = []
    if args.use_fp16:
        grad_scaler = GradScaler()
        plugins += [NativeMixedPrecisionPlugin(precision=16, device='cuda', scaler=grad_scaler)]

    wandb_logger = WandbLogger(project='glimpse_mae', entity="ideas_cv", reinit=True)

    checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{wandb.run.name}", save_top_k=3, monitor="val/loss")

    trainer = Trainer(plugins=plugins, max_epochs=args.epochs, accelerator='auto', logger=wandb_logger,
                      callbacks=[checkpoint_callback])

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
