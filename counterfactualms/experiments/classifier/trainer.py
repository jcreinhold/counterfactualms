import argparse
import logging
import os
import warnings
import sys

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from counterfactualms.experiments.classifier.classifier_experiment import ClassifierExperiment

logger = logging.getLogger(__name__)


def main():
    exp_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exp_parser.add_argument('--model-path', '-mp', help='pre-trained model to load',
                            default='/iacl/pg20/jacobr/calabresi/models/pretrained.ckpt', type=str)
    exp_parser.add_argument('--seed', default=1337, type=int, help='random seed')
    exp_parser.add_argument('-v', '--verbosity', action="count", default=0,
                            help="increase output verbosity (e.g., -vv is more than -v)")

    exp_args, other_args = exp_parser.parse_known_args()
    seed_everything(exp_args.seed)
    if exp_args.verbosity == 1:
        level = logging.getLevelName('INFO')
    elif exp_args.verbosity >= 2:
        level = logging.getLevelName('DEBUG')
    else:
        level = logging.getLevelName('WARNING')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(checkpoint_callback=True)

    parser._action_groups[1].title = 'lightning_options'

    experiment_group = parser.add_argument_group('experiment')
    ClassifierExperiment.add_arguments(experiment_group)

    args = parser.parse_args(other_args)

    if args.gpus is not None and isinstance(args.gpus, int):
        # Make sure that it only uses a single GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        args.gpus = 1

    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)

    lightning_args = groups['lightning_options']

    tb_logger = TensorBoardLogger(lightning_args.default_root_dir, name=f'ClassifierExperiment/')
    lightning_args.logger = tb_logger

    hparams = groups['experiment']
    setattr(hparams, 'n_epochs', args.max_epochs)
    setattr(hparams, 'verbosity', exp_args.verbosity)

    callbacks = [ModelCheckpoint(
        monitor='val_loss',
        save_top_k=10,
        save_last=True,
        mode='min',
        filename='{epoch}-{val_accuracy:.2f}-{val_loss:.2f}'
    )]
    trainer = Trainer.from_argparse_args(lightning_args, callbacks=callbacks)

    experiment = ClassifierExperiment(hparams)

    warning_level = "once" if hparams.validate else "ignore"
    with warnings.catch_warnings():
        warnings.simplefilter(warning_level)
        trainer.fit(experiment)


if __name__ == '__main__':
    sys.exit(main())
