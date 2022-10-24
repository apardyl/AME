import os
import random
import sys
import time
import torch
import logging

from utils.metrics import ReconstructionMetricsLogger
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_utils.datasets import ReconstructionDataset, create_train_val_datasets
from utils.train import parse_args, epoch_time, save_model, create_checkpoint_dir, dict_to_device


torch.set_printoptions(threshold=10_000)
random.seed(1)
torch.manual_seed(1)


def train_epoch(train_loader, device, model, args, grad_scaler):
    metrics_logger = ReconstructionMetricsLogger("train", args)
    model.train()
    for batch in tqdm(train_loader, desc="epoch"):
        dict_to_device(batch, device)
        model.module.optimizer.zero_grad()
        with autocast(enabled=args.use_fp16):
            out = model(batch)
            loss, loss_dict = model.module.calculate_loss(out, batch)
        if args.use_fp16:
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            grad_scaler.step(model.module.optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            model.module.optimizer.step()
        metrics_logger.log(out, batch, loss_dict)
        model.module.on_iter_end()
    return metrics_logger


def eval_epoch(loader, device, model, args):
    metrics_logger = ReconstructionMetricsLogger("valid", args)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="epoch"):
            dict_to_device(batch, device)

            with autocast(enabled=args.use_fp16):
                out = model(batch)
                loss, loss_dict = model.module.calculate_loss(out, batch)

            metrics_logger.log(out, batch, loss_dict)
    return metrics_logger


def main():
    # Parse args and prepare logger
    args = parse_args()
    args.checkpoint_dir = create_checkpoint_dir(args.checkpoint_dir)

    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.checkpoint_dir, "log.txt"))
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = args.arch(args)
    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))
    model = torch.nn.DataParallel(model)

    grad_scaler = GradScaler()

    # Initialize dataloaders
    train_dataset, valid_dataset = create_train_val_datasets(ReconstructionDataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=ReconstructionDataset.custom_collate, num_workers=args.num_workers,
                              sampler=RandomSampler(train_dataset, replacement=True, num_samples=args.num_samples))
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=ReconstructionDataset.custom_collate,
                              shuffle=False, num_workers=args.num_workers)

    logging.info((vars(args)))
    logging.info("Loaded {} train samples".format(len(train_dataset)))
    logging.info("Loaded {} valid samples".format(len(valid_dataset)))
    logging.info(model)
    logging.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters\n')
    logging.info(f'The model has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} frozen parameters\n')

    tb_writer = SummaryWriter(args.checkpoint_dir)
    tb_writer.add_text("args", str(args))
    best_val_metric, best_epoch = 999, 0
    for epoch in range(args.epochs):
        start_time = time.time()

        train_metrics = train_epoch(train_loader, device, model, args, grad_scaler).get_epoch_result(tb_writer, epoch)
        val_metrics = eval_epoch(valid_loader, device, model, args).get_epoch_result(tb_writer, epoch)

        save_model(model.module, args.checkpoint_dir, val_metrics["RMSE"] < best_val_metric)
        if val_metrics["RMSE"] < best_val_metric:
            best_val_metric = val_metrics["RMSE"]
            best_epoch = epoch

        epoch_mins, epoch_secs = epoch_time(start_time, time.time())
        model.module.on_epoch_end()

        logging.info('Epoch: {} | Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
        logging.info(ReconstructionMetricsLogger.result_to_string(train_metrics, "Train "))
        logging.info(ReconstructionMetricsLogger.result_to_string(val_metrics, "Valid "))
        logging.info("\n")

        if epoch - best_epoch >= args.patience:
            logging.info("Early stopping")
            break

    logging.info(f"Best epoch: {best_epoch}")


if __name__ == "__main__":
    main()
