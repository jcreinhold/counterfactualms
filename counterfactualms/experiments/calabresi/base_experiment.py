from typing import Tuple
from functools import partial
import logging
import os

import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from torch.utils.data import DataLoader

import pyro
from pyro.nn import PyroModule, pyro_method
from pyro.distributions import TransformedDistribution  # noqa: F401
from pyro.infer.reparam.transform import TransformReparam
from torch.distributions import Independent
from pyro.distributions.transforms import ComposeTransform, SigmoidTransform, AffineTransform

from counterfactualms.datasets.calabresi import CalabresiDataset

logger = logging.getLogger(__name__)

EXPERIMENT_REGISTRY = {}
MODEL_REGISTRY = {}


class BaseSEM(PyroModule):
    def __init__(self, preprocessing:str='realnvp', resize:Tuple[int,int]=(0,0)):
        super().__init__()
        self.resize = resize
        self.preprocessing = preprocessing

    def _get_preprocess_transforms(self):
        alpha = 0.05
        num_bits = 8
        if self.preprocessing == 'glow':
            # Map to [-0.5,0.5]
            a1 = AffineTransform(-0.5, (1. / 2 ** num_bits))
            preprocess_transform = ComposeTransform([a1])
        elif self.preprocessing == 'realnvp':
            # Map to [0,1]
            a1 = AffineTransform(0., (1. / 2 ** num_bits))
            # Map into unconstrained space as done in RealNVP
            a2 = AffineTransform(alpha, (1 - alpha))
            s = SigmoidTransform()
            preprocess_transform = ComposeTransform([a1, a2, s.inv])
        else:
            raise ValueError(f'{self.preprocessing} not valid.')
        return preprocess_transform

    @pyro_method
    def pgm_model(self):
        raise NotImplementedError()

    @pyro_method
    def model(self):
        raise NotImplementedError()

    @pyro_method
    def pgm_scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None
        return pyro.poutine.reparam(self.pgm_model, config=config)(*args, **kwargs)

    @pyro_method
    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution) and 'z' not in msg['name']:
                return TransformReparam()
            else:
                return None
        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    @pyro_method
    def sample(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.model()
        return samples

    @pyro_method
    def sample_scm(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.scm()
        return samples

    @pyro_method
    def infer_exogenous(self, obs):
        # assuming that we use transformed distributions for everything
        cond_sample = pyro.condition(self.sample, data=obs)
        batch_size = obs['x'].shape[0]
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(batch_size)

        output = {}
        for name, node in cond_trace.nodes.items():
            if 'z' in name or 'fn' not in node.keys():
                continue

            fn = node['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
            if isinstance(fn, TransformedDistribution):
                # compute exogenous base distribution at all sites. base dist created with TransformReparam
                output[name + '_base'] = ComposeTransform(fn.transforms).inv(node['value'])

        return output

    @pyro_method
    def infer(self, obs):
        raise NotImplementedError()

    @pyro_method
    def counterfactual(self, obs, condition=None):
        raise NotImplementedError()

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--preprocessing', default='realnvp', type=str, help="type of preprocessing (default: %(default)s)", choices=['realnvp', 'glow'])
        parser.add_argument('--resize', default=(128,128), type=int, nargs=2, help="resize cropped image to this size (use 0,0 for no resize) (default: %(default)s)")
        return parser


class BaseCovariateExperiment(pl.LightningModule):
    def __init__(self, hparams, pyro_model:BaseSEM):
        super().__init__()
        self.pyro_model = pyro_model
        hparams.experiment = self.__class__.__name__
        hparams.model = pyro_model.__class__.__name__
        self.hparams = hparams
        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size

        if hasattr(hparams, 'num_sample_particles'):
            self.pyro_model._gen_counterfactual = partial(self.pyro_model.counterfactual, num_particles=self.hparams.num_sample_particles)
        else:
            self.pyro_model._gen_counterfactual = self.pyro_model.counterfactual

        if hparams.validate:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.autograd.set_detect_anomaly(self.hparams.validate)
            pyro.enable_validation()

        resize = None if self.hparams.resize == (0,0) else self.hparams.resize
        train_crop_type = self.hparams.train_crop_type if hasattr(self.hparams, 'train_crop_type') else 'random'
        crop_size = self.hparams.crop_size if hasattr(self.hparams, 'crop_size') else (224, 224)
        self.calabresi_train = CalabresiDataset(self.hparams.train_csv, crop_size=crop_size, crop_type=train_crop_type, resize=resize)  # noqa: E501
        self.calabresi_val = CalabresiDataset(self.hparams.valid_csv, crop_size=crop_size, crop_type='center', resize=resize)
        self.calabresi_test = CalabresiDataset(self.hparams.test_csv, crop_size=crop_size, crop_type='center', resize=resize)

    def prepare_data(self):
        self.torch_device = self.trainer.root_gpu if self.trainer.on_gpu else self.trainer.root_device

        age = torch.from_numpy(self.calabresi_train.csv['age'].to_numpy())
        self.pyro_model.age_flow_lognorm_loc = age.log().mean().to(self.torch_device).float()
        self.pyro_model.age_flow_lognorm_scale = age.log().std().to(self.torch_device).float()

        duration = torch.from_numpy(self.calabresi_train.csv['duration'].to_numpy())
        self.pyro_model.duration_flow_lognorm_loc = duration.log().mean().to(self.torch_device).float()
        self.pyro_model.duration_flow_lognorm_scale = duration.log().std().to(self.torch_device).float()

        edss = torch.from_numpy(self.calabresi_train.csv['edss'].to_numpy())
        self.pyro_model.edss_flow_lognorm_loc = edss.log().mean().to(self.torch_device).float()
        self.pyro_model.edss_flow_lognorm_scale = edss.log().std().to(self.torch_device).float()

        ventricle_volume = torch.from_numpy(self.calabresi_train.csv['ventricle_volume'].to_numpy())
        self.pyro_model.ventricle_volume_flow_lognorm_loc = ventricle_volume.log().mean().to(self.torch_device).float()
        self.pyro_model.ventricle_volume_flow_lognorm_scale = ventricle_volume.log().std().to(self.torch_device).float()

        brain_volume = torch.from_numpy(self.calabresi_train.csv['brain_volume'].to_numpy())
        self.pyro_model.brain_volume_flow_lognorm_loc = brain_volume.log().mean().to(self.torch_device).float()
        self.pyro_model.brain_volume_flow_lognorm_scale = brain_volume.log().std().to(self.torch_device).float()

        lesion_volume = torch.from_numpy(self.calabresi_train.csv['lesion_volume'].to_numpy())
        self.pyro_model.lesion_volume_flow_lognorm_loc = lesion_volume.log().mean().to(self.torch_device).float()
        self.pyro_model.lesion_volume_flow_lognorm_scale = lesion_volume.log().std().to(self.torch_device).float()

        slice_number = torch.from_numpy(self.calabresi_train.csv['slice_number'].to_numpy())
        self.pyro_model.slice_number_min = torch.tensor([slice_number.min()-0.5-1e-3], device=self.torch_device, dtype=torch.float32)
        self.pyro_model.slice_number_max = torch.tensor([slice_number.max()+0.5+1e-3], device=self.torch_device, dtype=torch.float32)

        perm = lambda: torch.randperm(self.pyro_model.latent_dim, dtype=torch.long, requires_grad=False).to(self.torch_device)
        if self.pyro_model.use_prior_permutations:
            for i in range(self.pyro_model.n_prior_flows):
                setattr(self.pyro_model, f'prior_flow_permutation_{i}', perm())
        if self.pyro_model.use_posterior_permutations:
            for i in range(self.pyro_model.n_posterior_flows):
                setattr(self.pyro_model, f'posterior_flow_permutation_{i}', perm())

        if self.hparams.validate:
            logger.info(f'set age_flow_lognorm {self.pyro_model.age_flow_lognorm.loc} +/- {self.pyro_model.age_flow_lognorm.scale}')
            logger.info(f'set brain_volume_flow_lognorm {self.pyro_model.brain_volume_flow_lognorm.loc} +/- {self.pyro_model.brain_volume_flow_lognorm.scale}')
            logger.info(f'set ventricle_volume_flow_lognorm {self.pyro_model.ventricle_volume_flow_lognorm.loc} +/- {self.pyro_model.ventricle_volume_flow_lognorm.scale}')  # noqa: E501
            logger.info(f'set lesion_volume_flow_lognorm {self.pyro_model.lesion_volume_flow_lognorm.loc} +/- {self.pyro_model.lesion_volume_flow_lognorm.scale}')  # noqa: E501
            logger.info(f'set duration_flow_lognorm {self.pyro_model.duration_flow_lognorm.loc} +/- {self.pyro_model.duration_flow_lognorm.scale}')  # noqa: E501
            logger.info(f'set edss_flow_lognorm {self.pyro_model.edss_flow_lognorm.loc} +/- {self.pyro_model.edss_flow_lognorm.scale}')  # noqa: E501
            logger.info(f'set slice min/max {int(self.pyro_model.slice_number_min.item()):d} / {int(self.pyro_model.slice_number_max.item()):d}')  # noqa: E501

    def configure_optimizers(self):
        pass

    def _dataloader_params(self):
        num_workers = len(os.sched_getaffinity(0)) // 2  # use half of the available cpus
        return {'num_workers': num_workers, 'pin_memory': self.trainer.on_gpu}

    def train_dataloader(self):
        return DataLoader(self.calabresi_train, batch_size=self.train_batch_size,
                          shuffle=True, drop_last=True, **self._dataloader_params())

    def val_dataloader(self):
        return DataLoader(self.calabresi_val, batch_size=self.test_batch_size,
                          shuffle=False, **self._dataloader_params())

    def test_dataloader(self):
        return DataLoader(self.calabresi_test, batch_size=self.test_batch_size,
                          shuffle=False, **self._dataloader_params())

    def forward(self, *args, **kwargs):
        pass

    def prep_batch(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_epoch_end(self, outputs):
        if self.current_epoch % self.hparams.sample_img_interval == 0:
            self.sample_images()

    def test_epoch_end(self, outputs):
        metrics = outputs['metrics']
        samples = outputs['samples']
        sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)
        samples['unconditional_samples'] = {k: sample_trace.nodes[k]['value'].cpu() for k in self.required_data}

        logger.info(f'Got samples: {tuple(samples.keys())}')
        for k, v in samples.items():
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{k}.pt')
            logging.info(f'Saving samples for {k} to {p}')
            torch.save(v, p)

        p = os.path.join(self.trainer.logger.experiment.log_dir, 'metrics.pt')
        torch.save(metrics, p)

    def assemble_epoch_end_outputs(self, outputs):
        num_items = len(outputs)

        def handle_row(batch, assembled=None):
            if assembled is None:
                assembled = {}

            for k, v in batch.items():
                if k not in assembled.keys():
                    if isinstance(v, dict):
                        assembled[k] = handle_row(v)
                    elif isinstance(v, float):
                        assembled[k] = v
                    elif np.prod(v.shape) == 1:
                        assembled[k] = v.cpu()
                    else:
                        assembled[k] = v.cpu()
                else:
                    if isinstance(v, dict):
                        assembled[k] = handle_row(v, assembled[k])
                    elif isinstance(v, float):
                        assembled[k] += v
                    elif np.prod(v.shape) == 1:
                        assembled[k] += v.cpu()
                    else:
                        assembled[k] = torch.cat([assembled[k], v.cpu()], 0)

            return assembled

        assembled = {}
        for _, batch in enumerate(outputs):
            assembled = handle_row(batch, assembled)

        for k, v in assembled.items():
            if (hasattr(v, 'shape') and np.prod(v.shape) == 1) or isinstance(v, float):
                assembled[k] /= num_items

        return assembled

    def get_counterfactual_conditions(self, batch):
        counterfactuals = {
            'do(brain_volume=8000)':  {'brain_volume': torch.ones_like(batch['brain_volume']) * 8000},
            'do(brain_volume=16000)': {'brain_volume': torch.ones_like(batch['brain_volume']) * 16000},
            'do(ventricle_volume=500)':  {'ventricle_volume': torch.ones_like(batch['ventricle_volume']) * 500},
            'do(ventricle_volume=2000)': {'ventricle_volume': torch.ones_like(batch['ventricle_volume']) * 2000},
            'do(lesion_volume=0)':    {'lesion_volume': torch.ones_like(batch['lesion_volume']) * 1e-5},
            'do(lesion_volume=1000)': {'lesion_volume': torch.ones_like(batch['lesion_volume']) * 1000.},
            'do(age=20)': {'age': torch.ones_like(batch['age']) * 20},
            'do(age=60)': {'age': torch.ones_like(batch['age']) * 60},
            'do(sex=0)': {'sex': torch.zeros_like(batch['sex'])},
            'do(sex=1)': {'sex': torch.ones_like(batch['sex'])},
            'do(duration=0)': {'duration': torch.zeros_like(batch['duration']) + 1e-5},
            'do(duration=12)': {'duration': torch.ones_like(batch['duration']) * 12.},
            'do(edss=0)': {'edss': torch.ones_like(batch['edss']) + 1e-5},
            'do(edss=6)': {'edss': torch.ones_like(batch['edss']) * 6.},
            'do(brain_volume=8000, ventricle_volume=500)': {'brain_volume': torch.ones_like(batch['brain_volume']) * 8000.,
                                                            'ventricle_volume': torch.ones_like(batch['ventricle_volume']) * 500.},
            'do(brain_volume=16000, ventricle_volume=1000)': {'brain_volume': torch.ones_like(batch['brain_volume']) * 16000.,
                                                              'ventricle_volume': torch.ones_like(batch['ventricle_volume']) * 1000.}
        }
        return counterfactuals

    def build_test_samples(self, batch):
        samples = {}
        samples['reconstruction'] = {'x': self.pyro_model.reconstruct(batch, num_particles=self.hparams.num_sample_particles)}

        counterfactuals = self.get_counterfactual_conditions(batch)

        for name, condition in counterfactuals.items():
            samples[name] = self.pyro_model._gen_counterfactual(obs=batch, condition=condition)

        return samples

    def log_img_grid(self, tag, imgs, normalize=True, save_img=False, **kwargs):
        imgs.clamp_(min=0., max=255.)
        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            torchvision.utils.save_image(imgs, p)
        grid = torchvision.utils.make_grid(imgs, normalize=normalize, **kwargs)
        self.logger.experiment.add_image(tag, grid, self.current_epoch)

    def get_batch(self, loader):
        batch = next(iter(loader))
        batch = {k: v.to(self.torch_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    def log_kdes(self, tag, data, save_img=False):
        def np_val(x):
            return x.cpu().numpy().squeeze() if isinstance(x, torch.Tensor) else x.squeeze()

        fig, ax = plt.subplots(1, len(data), figsize=(5 * len(data), 3), sharex=True, sharey=True)
        for i, (name, covariates) in enumerate(data.items()):
            try:
                if len(covariates) == 1:
                    (x_n, x), = tuple(covariates.items())
                    sns.histplot(x=np_val(x), ax=ax[i])
                elif len(covariates) == 2:
                    (x_n, x), (y_n, y) = tuple(covariates.items())
                    sns.histplot(x=np_val(x), y=np_val(y), ax=ax[i])
                    ax[i].set_ylabel(y_n)
                else:
                    raise ValueError(f'got too many values: {len(covariates)}')
            except Exception as e:
                logging.info(e)
                logging.info(f'Got the above error when plotting {tag}/{name}')

            ax[i].set_title(name)
            ax[i].set_xlabel(x_n)

        sns.despine()

        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            plt.savefig(p, dpi=300)

        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

    def build_reconstruction(self, obs, tag='recon'):
        self._check_observation(obs)
        x = obs['x']
        recon = self.pyro_model.reconstruct(obs, num_particles=self.hparams.num_sample_particles)
        self.log_img_grid(tag+'_img', torch.cat([x, recon], 0))
        mse = torch.mean(torch.square(x - recon).sum((1, 2, 3)))
        self.log(f'{tag}', mse)

    @property
    def required_data(self):
        return {'x', 'sex', 'age', 'ventricle_volume', 'brain_volume', 'lesion_volume',
                'edss', 'duration', 'slice_number'}

    def _check_observation(self, obs):
        keys = obs.keys()
        assert self.required_data == set(keys), f'Incompatible observation: {tuple(keys)}'

    def build_counterfactual(self, tag, obs, conditions, absolute=None, kde=False):
        self._check_observation(obs)
        imgs = [obs['x']]
        if kde:
            if absolute == 'brain_volume':
                sampled_kdes = {'orig': {'ventricle_volume': obs['ventricle_volume']}}
            elif absolute == 'ventricle_volume':
                sampled_kdes = {'orig': {'brain_volume': obs['brain_volume']}}
            else:
                sampled_kdes = {'orig': {'brain_volume': obs['brain_volume'],
                                         'ventricle_volume': obs['ventricle_volume']}}

        for name, data in conditions.items():
            counterfactual = self.pyro_model._gen_counterfactual(obs=obs, condition=data)

            imgs.append(counterfactual['x'])

            if kde:
                if absolute == 'brain_volume':
                    sampled_kdes[name] = {'ventricle_volume': counterfactual['ventricle_volume']}
                elif absolute == 'ventricle_volume':
                    sampled_kdes[name] = {'brain_volume': counterfactual['brain_volume']}
                else:
                    sampled_kdes[name] = {'brain_volume': counterfactual['brain_volume'],
                                          'ventricle_volume': counterfactual['ventricle_volume']}

        self.log_img_grid(tag, torch.cat(imgs, 0))
        if kde:
            self.log_kdes(f'{tag}_sampled', sampled_kdes, save_img=True)

    def sample_images(self):
        with torch.no_grad():
            sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)

            samples = sample_trace.nodes['x']['value']
            sampled_brain_volume = sample_trace.nodes['brain_volume']['value']
            sampled_ventricle_volume = sample_trace.nodes['ventricle_volume']['value']

            s = samples.shape[0] // 8
            m = 8 * s
            self.log_img_grid('samples', samples.data[:m:s])

            obs_batch = self.prep_batch(self.get_batch(self.val_dataloader()))

            kde_data = {
                'batch': {'brain_volume': obs_batch['brain_volume'],
                          'ventricle_volume': obs_batch['ventricle_volume']},
                'sampled': {'brain_volume': sampled_brain_volume,
                            'ventricle_volume': sampled_ventricle_volume}
            }
            self.log_kdes('sample_kde', kde_data, save_img=True)

            exogenous = self.pyro_model.infer(obs_batch)

            for (tag, val) in exogenous.items():
                self.logger.experiment.add_histogram(tag, val, self.current_epoch)

            s = obs_batch['x'].shape[0] // 8
            m = 8 * s
            obs_batch = {k: v[:m:s] for k, v in obs_batch.items()}

            self.log_img_grid('input', obs_batch['x'], save_img=True)

            if hasattr(self.pyro_model, 'reconstruct'):
                self.build_reconstruction(obs_batch)

            conditions = {
                '20': {'age': torch.zeros_like(obs_batch['age']) + 20},
                '60': {'age': torch.zeros_like(obs_batch['age']) + 60},
            }
            self.build_counterfactual('do(age=x)', obs=obs_batch, conditions=conditions)

            conditions = {
                '0': {'sex': torch.zeros_like(obs_batch['sex'])},
                '1': {'sex': torch.ones_like(obs_batch['sex'])},
            }
            self.build_counterfactual('do(sex=x)', obs=obs_batch, conditions=conditions)

            conditions = {
                '1000000': {'brain_volume': torch.zeros_like(obs_batch['brain_volume']) + 1000000},
                '1600000': {'brain_volume': torch.zeros_like(obs_batch['brain_volume']) + 1800000}
            }
            self.build_counterfactual('do(brain_volume=x)', obs=obs_batch, conditions=conditions)

            conditions = {
                '15000':   {'ventricle_volume': torch.zeros_like(obs_batch['ventricle_volume']) + 15000},
                '50000': {'ventricle_volume': torch.zeros_like(obs_batch['ventricle_volume']) + 50000},
            }
            self.build_counterfactual('do(ventricle_volume=x)', obs=obs_batch, conditions=conditions)

            conditions = {
                '0':    {'lesion_volume': torch.zeros_like(obs_batch['lesion_volume']) + 1e-5},
                '60000': {'lesion_volume': torch.zeros_like(obs_batch['lesion_volume']) + 60000},
            }
            self.build_counterfactual('do(lesion_volume=x)', obs=obs_batch, conditions=conditions)

            conditions = {
                '1': {'edss': torch.zeros_like(obs_batch['edss']) + 1.},
                '5': {'edss': torch.zeros_like(obs_batch['edss']) + 5.}
            }
            self.build_counterfactual('do(edss=x)', obs=obs_batch, conditions=conditions)

            conditions = {
                '0': {'duration': torch.zeros_like(obs_batch['duration']) + 1e-5},
                '10': {'duration': torch.zeros_like(obs_batch['duration']) + 10.},
            }
            self.build_counterfactual('do(duration=x)', obs=obs_batch, conditions=conditions)

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--train-csv', default="/iacl/pg20/jacobr/calabresi/png/csv/train_png.csv", type=str, help="csv for training data (default: %(default)s)")  # noqa: E501
        parser.add_argument('--valid-csv', default="/iacl/pg20/jacobr/calabresi/png/csv/valid_png.csv", type=str, help="csv for validation data (default: %(default)s)")  # noqa: E501
        parser.add_argument('--test-csv', default="/iacl/pg20/jacobr/calabresi/png/csv/test_png.csv", type=str, help="csv for testing data (default: %(default)s)")  # noqa: E501
        parser.add_argument('--crop-size', default=(224,224), type=int, nargs=2, help="size of patch to take from image (default: %(default)s)")
        parser.add_argument('--train-crop-type', default='random', choices=['random', 'center'], help="how to crop training images (default: %(default)s)")
        parser.add_argument('--sample-img-interval', default=5, type=int, help="interval in which to sample and log images (default: %(default)s)")
        parser.add_argument('--train-batch-size', default=128, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--test-batch-size', default=64, type=int, help="test batch size (default: %(default)s)")
        parser.add_argument('--validate', default=False, action='store_true', help="whether to validate (default: %(default)s)")
        parser.add_argument('--lr', default=1e-3, type=float, help="lr of deep part (default: %(default)s)")
        parser.add_argument('--pgm-lr', default=5e-3, type=float, help="lr of pgm (default: %(default)s)")
        parser.add_argument('--weight-decay', default=0., type=float, help="weight decay for adam (default: %(default)s)")
        parser.add_argument('--betas', default=(0.9,0.999), type=float, nargs=2, help="betas for adam (default: %(default)s)")
        parser.add_argument('--clip-norm', default=100., type=float, help="clip norm for grad for adam (default: %(default)s)")
        parser.add_argument('--pgm-clip-norm', default=10., type=float, help="clip norm for grad for adam in pgm (default: %(default)s)")
        parser.add_argument('--flow-clip-norm', default=1., type=float, help="clip norm for grad for adam in prior/posterior flow (default: %(default)s)")
        parser.add_argument('--use-exponential-lr', default=False, action='store_true', help="use exponential lr decay, else use 1cycle (default: %(default)s)")
        parser.add_argument('--lrd', default=0.9999, type=float, help="learning rate decay w/ exponential (default: %(default)s)")
        parser.add_argument('--pct-start', default=0.1, type=float, help="pct of training epochs to ramp up lr (default: %(default)s)")
        parser.add_argument('--div-factor', default=25., type=float, help="div factor from lr for initial lr in 1cycle (default: %(default)s)")
        parser.add_argument('--final-div-factor', default=1e4, type=float, help="div factor from lr for final lr in 1cycle (default: %(default)s)")
        parser.add_argument('--use-adagrad-rmsprop', default=False, action='store_true', help="use adagrad-rmsprop mashup (default: %(default)s)")
        parser.add_argument('--eta', default=1.0, type=float, help="eta in adagrad-rmsprop mashup (default: %(default)s)")
        parser.add_argument('--delta', default=1e-16, type=float, help="delta in adagrad-rmsprop mashup (default: %(default)s)")
        parser.add_argument('--t', default=0.1, type=float, help="t in adagrad-rmsprop mashup (default: %(default)s)")
        return parser
