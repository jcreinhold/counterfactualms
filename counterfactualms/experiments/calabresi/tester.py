import argparse
from collections import OrderedDict
import inspect
import logging
import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.parsing import AttributeDict
import torch

from counterfactualms.experiments import calabresi  # noqa: F401
from counterfactualms.experiments.calabresi.base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY

logger = logging.getLogger(__name__)
_buffers_to_load = ('norm', 'permutation', 'slice_number')


def main():
    exp_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exp_parser.add_argument('--checkpoint-path', '-c', type=str, help='which checkpoint to load')
    exp_parser.add_argument('--batch-size', '-bs', type=int, help='batch size')
    exp_parser.add_argument('--resize', '-rs', nargs=2, type=int, help='resize')
    exp_parser.add_argument('--out-dir', '-od', type=str, help='output directory for nifti counterfactuals')
    exp_parser.add_argument('--csv', '-csv', type=str, help='csv path')
    exp_parser.add_argument('-v', '--verbosity', action="count", default=0,
                            help="increase output verbosity (e.g., -vv is more than -v)")

    exp_args, other_args = exp_parser.parse_known_args()
    if exp_args.verbosity == 1:
        level = logging.getLevelName('INFO')
    elif exp_args.verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)

    checkpoint_path = exp_args.checkpoint_path
    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    hparams = ckpt['hyper_parameters']
    logger.info(f'found hparams: {hparams}')
    exp_class = EXPERIMENT_REGISTRY[hparams['experiment']]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(checkpoint_callback=True)
    parser._action_groups[1].title = 'lightning_options'
    args = parser.parse_args(other_args)

    if args.gpus is not None and isinstance(args.gpus, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        args.gpus = 1

    exp = hparams['model']
    model_class = MODEL_REGISTRY[exp]
    model_params = {
        k: v for k, v in hparams.items() if (k in inspect.signature(model_class.__init__).parameters
                                             or k in k in inspect.signature(model_class.__bases__[0].__init__).parameters
                                             or k in k in inspect.signature(model_class.__bases__[0].__bases__[0].__init__).parameters)
    }
    model_params['img_shape'] = hparams['resize'] if 'resize' in hparams else exp_args.resize
    new_state_dict = OrderedDict()
    for key, value in ckpt['state_dict'].items():
        new_key = key.replace('pyro_model.', '')
        new_state_dict[new_key] = value
    loaded_model = model_class(**model_params)
    loaded_model.load_state_dict(new_state_dict)
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpus is not None) else 'cpu')
    for p in loaded_model._buffers.keys():
        if any([(b in p) for b in _buffers_to_load]):
            setattr(loaded_model, p, getattr(loaded_model, p).to(device))
    loaded_model.eval()
    loaded_model = loaded_model.to(device)

    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    lightning_args = groups['lightning_options']
    trainer = Trainer.from_argparse_args(lightning_args)
    trainer.logger.experiment.log_dir = exp_args.checkpoint_path
    hparams = AttributeDict(hparams)
    hparams.test_dir = exp_args.out_dir
    hparams.test_csv = exp_args.csv
    hparams.test_batch_size = exp_args.batch_size
    hparams.test_batch_size = exp_args.batch_size
    experiment = exp_class.load_from_checkpoint(checkpoint_path, hparams=hparams, pyro_model=loaded_model)
    logger.info(f'Loaded {experiment.__class__}:\n{experiment}')
    trainer.test(experiment)


if __name__ == '__main__':
    sys.exit(main())

