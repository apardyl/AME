import sys

import torch
import tqdm

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
    parser.add_argument('--save-path',
                        help='path to save visualizations to',
                        type=str)
    parser.add_argument('--epochs',
                        help='number of epochs',
                        type=int,
                        default=50)
    return parent_parser


def main():
    data_module, model, args = experiment_from_args(sys.argv, add_argparse_args_fn=define_args)
    model.load_state_dict(torch.load(args.load_model_path, map_location='cpu')['state_dict'])

    assert isinstance(data_module, BaseReconstructionDataModule)

    data_module.setup('test')
    loader = data_module.test_dataloader()

    model = model.cuda()
    model.eval()
    model.debug = True

    for idx, batch in enumerate(tqdm.tqdm(loader, total=min(args.max_batches, len(loader)))):
        if idx >= args.max_batches:
            break
        out = model.predict_step([x.cuda() for x in batch], idx)
        save_reconstructions(model, out, batch, vis_id=idx, dump_path=args.save_path)


if __name__ == "__main__":
    main()
