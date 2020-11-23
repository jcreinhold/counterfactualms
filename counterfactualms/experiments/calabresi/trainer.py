import argparse
from collections import OrderedDict
import logging
import os
import warnings
import sys

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from counterfactualms.experiments import calabresi # noqa: F401
from counterfactualms.experiments.calabresi.base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY

logger = logging.getLogger(__name__)


def main():
    exp_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exp_parser.add_argument('--experiment', '-e', help='which experiment to load', choices=tuple(EXPERIMENT_REGISTRY.keys()))
    exp_parser.add_argument('--model', '-m', help='which model to load', choices=tuple(MODEL_REGISTRY.keys()))
    exp_parser.add_argument('--model-path', '-mp', help='pre-trained model to load',
                            default='/iacl/pg20/jacobr/calabresi/models/pretrained.ckpt', type=str)
    exp_parser.add_argument('--seed', default=1337, type=int, help='random seed')
    exp_parser.add_argument('-v', '--verbosity', action="count", default=0,
                            help="increase output verbosity (e.g., -vv is more than -v)")

    exp_args, other_args = exp_parser.parse_known_args()
    if exp_args.verbosity == 1:
        level = logging.getLevelName('INFO')
    elif exp_args.verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('ERROR')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)

    exp_class = EXPERIMENT_REGISTRY[exp_args.experiment]
    model_class = MODEL_REGISTRY[exp_args.model]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(checkpoint_callback=True)

    parser._action_groups[1].title = 'lightning_options'

    experiment_group = parser.add_argument_group('experiment')
    exp_class.add_arguments(experiment_group)

    model_group = parser.add_argument_group('model')
    model_class.add_arguments(model_group)

    args = parser.parse_args(other_args)

    if args.gpus is not None and isinstance(args.gpus, int):
        # Make sure that it only uses a single GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        args.gpus = 1

    # TODO: push to lightning
    args.gradient_clip_val = float(args.gradient_clip_val)

    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)

    lightning_args = groups['lightning_options']

    tb_logger = TensorBoardLogger(lightning_args.default_root_dir, name=f'{exp_args.experiment}/{exp_args.model}')
    lightning_args.logger = tb_logger

    hparams = groups['experiment']
    model_params = groups['model']

    for k, v in vars(model_params).items():
        setattr(hparams, k, v)

    trainer = Trainer.from_argparse_args(lightning_args)

    model_dict = vars(model_params)
    model_dict['img_shape'] = args.crop_size
    model = model_class(**model_dict)
    if exp_args.model_path is not None:
        ckpt = torch.load(exp_args.model_path, map_location=torch.device('cpu'))
        state_dict = ckpt['state_dict']
        model_state_dict = model.state_dict()
        new_state_dict = OrderedDict()
        for k in state_dict:
            new_key = k.replace('pyro_model.', '')
            if new_key in model_state_dict:
                if state_dict[k].shape != model_state_dict[new_key].shape:
                    logger.info(f"Skip loading parameter: {new_key}, "
                                f"required shape: {model_state_dict[new_key].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    new_state_dict[new_key] = model_state_dict[new_key]
            else:
                logger.info(f"Dropping parameter {k}")
        model.load_state_dict(new_state_dict, strict=False)
    experiment = exp_class(hparams, model)
    seed_everything(exp_args.seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(experiment)


if __name__ == '__main__':
    sys.exit(main())
