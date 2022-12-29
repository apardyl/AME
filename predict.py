import argparse
import os
import sys

import torch
import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
from pytorch_lightning.strategies import DDPStrategy

from architectures.reconstruction import ReconstructionMae
from utils.prepare import experiment_from_args
from utils.visualize import save_reconstructions


def define_args(parent_parser):
    parser = parent_parser.add_argument_group('predict.py')
    parser.add_argument('--load-model-path',
                        help='load model from pth',
                        type=str,
                        required=True)
    parser.add_argument('--max-batches',
                        help='number of batches from dataset to process',
                        type=int,
                        default=99999999999999)
    parser.add_argument('--visualization-path',
                        help='path to save visualizations to',
                        type=str)
    parser.add_argument('--ddp',
                        help='use DDP acceleration strategy',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--test',
                        help='do test',
                        type=bool,
                        default=False,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--split',
                        help='split to use',
                        choices=['train', 'val', 'test'],
                        type=str,
                        default=None)
    parser.add_argument('--dump-path',
                        help='do latent dump to file',
                        type=str,
                        default=None)
    return parent_parser


def do_visualizations(args, model, loader):
    model.load_state_dict(torch.load(args.load_model_path, map_location='cpu')['state_dict'])

    model = model.cuda()
    model.eval()
    model.debug = True

    assert isinstance(model, ReconstructionMae)

    for idx, batch in enumerate(tqdm.tqdm(loader, total=min(args.max_batches, len(loader)))):
        if idx >= args.max_batches:
            break
        out = model.predict_step([x.cuda() for x in batch], idx)
        save_reconstructions(model, out, batch, vis_id=idx, dump_path=args.visualization)

    model.debug = False


def do_dump_latent(args, model, loader):
    os.makedirs(args.dump_path, exist_ok=True)
    model.load_pretrained_mae(args.load_model_path)

    model = model.cuda()
    model.eval()
    model.debug = True

    latents = []
    targets = []

    for idx, batch in enumerate(tqdm.tqdm(loader, total=min(args.max_batches, len(loader)))):
        if idx >= args.max_batches:
            break
        out = model.predict_step([x.cuda() for x in batch], idx)
        latents.append(out['steps'][-1]['latent'])
        targets.append(batch[1])
    latents = torch.cat(latents, dim=0)
    targets = torch.cat(targets, dim=0)
    torch.save({'latents': latents, 'targets': targets}, os.path.join(args.dump_path, f'embeds_{args.split}.pck'))

    model.debug = False


def do_test(args, model, loader):
    trainer = Trainer(accelerator='auto', callbacks=[RichProgressBar(leave=True), RichModelSummary(max_depth=3)],
                      strategy=DDPStrategy(find_unused_parameters=False) if args.ddp else None)
    trainer.test(model=model, ckpt_path=args.load_model_path, dataloaders=loader)


def main():
    data_module, model, args = experiment_from_args(sys.argv, add_argparse_args_fn=define_args)

    split = args.split
    if split is None:
        if data_module.has_test_data:
            split = 'test'
        else:
            split = 'val'
    if split == 'test':
        data_module.setup('test')
    else:
        data_module.setup('fit')
    args.split = split
    loader = {
        'train': data_module.train_dataloader,
        'val': data_module.val_dataloader,
        'test': data_module.test_dataloader
    }[split](no_aug=True)

    if args.visualization_path is not None:
        do_visualizations(args, model, loader)

    if args.test:
        do_test(args, model, loader)

    if args.dump_path is not None:
        do_dump_latent(args, model, loader)


if __name__ == "__main__":
    main()
