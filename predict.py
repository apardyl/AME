import argparse
import sys

import torch
import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
from pytorch_lightning.strategies import DDPStrategy

from datasets.reconstruction import BaseReconstructionDataModule
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
                        default=100)
    parser.add_argument('--visualization-path',
                        help='path to save visualizations to',
                        type=str)
    parser.add_argument('--epochs',
                        help='number of epochs',
                        type=int,
                        default=50)
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
    return parent_parser


def do_visualizations(args, model, data_module):
    model.load_state_dict(torch.load(args.load_model_path, map_location='cpu')['state_dict'])

    assert isinstance(data_module, BaseReconstructionDataModule)

    if data_module.has_test_data:
        data_module.setup('test')
        loader = data_module.test_dataloader()
    else:
        data_module.setup('fit')
        loader = data_module.val_dataloader()

    model = model.cuda()
    model.eval()
    model.debug = True

    for idx, batch in enumerate(tqdm.tqdm(loader, total=min(args.max_batches, len(loader)))):
        if idx >= args.max_batches:
            break
        out = model.predict_step([x.cuda() for x in batch], idx)
        save_reconstructions(model, out, batch, vis_id=idx, dump_path=args.visualization)

    model.debug = False


def do_test(args, model, data_module):
    trainer = Trainer(accelerator='auto', callbacks=[RichProgressBar(leave=True), RichModelSummary(max_depth=3)],
                      strategy=DDPStrategy(find_unused_parameters=False) if args.ddp else None)
    trainer.test(model=model, ckpt_path=args.load_model_path, datamodule=data_module)


def main():
    data_module, model, args = experiment_from_args(sys.argv, add_argparse_args_fn=define_args)

    if args.visualization:
        do_visualizations(args, model, data_module)

    if args.test:
        do_test(args, model, data_module)


if __name__ == "__main__":
    main()
