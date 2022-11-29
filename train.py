import logging
import os
import random
import sys

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, RandomSampler

from config import TRAIN_PATH, VALID_PATH
from data_utils.datasets import ReconstructionDataset, create_train_val_datasets
from utils.train import parse_args, create_checkpoint_dir

random.seed(1)
torch.manual_seed(1)


def main():
    # Parse args and prepare logger
    args = parse_args()
    logging.info((vars(args)))
    args.checkpoint_dir = create_checkpoint_dir(args.checkpoint_dir)

    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.checkpoint_dir, "log.txt"))
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    model = args.arch(args)
    model.load_pretrained_mae()
    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))

    logging.info(model)
    logging.info(
        f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters\n')
    logging.info(
        f'The model has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} frozen parameters\n')

    train_dataset, valid_dataset = create_train_val_datasets(ReconstructionDataset, TRAIN_PATH, VALID_PATH)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              sampler=RandomSampler(train_dataset, replacement=True, num_samples=args.num_samples))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    logging.info("Loaded {} train samples".format(len(train_dataset)))
    logging.info("Loaded {} valid samples".format(len(valid_dataset)))

    plugins = []
    if args.use_fp16:
        grad_scaler = GradScaler()
        plugins += [NativeMixedPrecisionPlugin(precision=16, device='cuda', scaler=grad_scaler)]

    wandb_logger = WandbLogger(project='glimpse_mae', entity="ideas_cv", mode='disabled')

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", save_top_k=3, monitor="val/loss")

    trainer = Trainer(plugins=plugins, max_epochs=args.epochs, accelerator='auto', logger=wandb_logger,
                      callbacks=[checkpoint_callback])

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    main()
