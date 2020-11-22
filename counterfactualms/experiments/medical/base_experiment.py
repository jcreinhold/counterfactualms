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

from counterfactualms.datasets.medical.calabresi import CalabresiDataset

logger = logging.getLogger(__name__)

EXPERIMENT_REGISTRY = {}
MODEL_REGISTRY = {}


class BaseSEM(PyroModule):
    def __init__(self, preprocessing:str='realnvp', downsample:int=-1):
        super().__init__()
        self.downsample = downsample
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
            if isinstance(msg['fn'], TransformedDistribution):
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
    def infer_e_x(self, *args, **kwargs):
        raise NotImplementedError()

    @pyro_method
    def infer_exogeneous(self, obs):
        # assuming that we use transformed distributions for everything:
        cond_sample = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(obs['x'].shape[0])

        output = {}
        for name, node in cond_trace.nodes.items():
            if 'fn' not in node.keys():
                continue

            fn = node['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
            if isinstance(fn, TransformedDistribution):
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
        parser.add_argument('--downsample', default=3, type=int, help="downsampling factor (-1 for none) (default: %(default)s)")
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
            pl.seed_everything(1337)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.autograd.set_detect_anomaly(self.hparams.validate)
            pyro.enable_validation()

    def prepare_data(self):
        downsample = None if self.hparams.downsample == -1 else self.hparams.downsample
        train_crop_type = self.hparams.train_crop_type if hasattr(self.hparams, 'train_crop_type') else 'random'
        crop_size = self.hparams.crop_size if hasattr(self.hparams, 'crop_size') else (192, 192)
        self.calabresi_train = CalabresiDataset(self.hparams.train_csv, crop_size=crop_size, crop_type=train_crop_type, downsample=downsample)  # noqa: E501
        self.calabresi_val = CalabresiDataset(self.hparams.valid_csv, crop_size=crop_size, crop_type='center', downsample=downsample)
        self.calabresi_test = CalabresiDataset(self.hparams.test_csv, crop_size=crop_size, crop_type='center', downsample=downsample)

        self.torch_device = self.trainer.root_gpu if self.trainer.on_gpu else self.trainer.root_device

        slice_brain_volumes = torch.linspace(8000., 16000., 3, dtype=torch.float, device=self.torch_device)
        self.slice_brain_volume_range = slice_brain_volumes.repeat(3).unsqueeze(1)
        slice_ventricle_volumes = torch.linspace(10., 2500., 3, dtype=torch.float, device=self.torch_device)
        self.slice_ventricle_volume_range = slice_ventricle_volumes.repeat_interleave(3).unsqueeze(1)
        slice_lesion_volumes = torch.linspace(1e-5, 2000., 3, dtype=torch.float, device=self.torch_device)
        self.slice_lesion_volume_range = slice_lesion_volumes.repeat_interleave(3).unsqueeze(1)
        scores = torch.arange(-1., 6., dtype=torch.float, device=self.torch_device)
        self.score_range = scores.repeat_interleave(3).unsqueeze(1)
        self.z_range = torch.randn([1, self.hparams.latent_dim], dtype=torch.float, device=self.torch_device).repeat((9, 1))

        age = torch.from_numpy(self.calabresi_train.csv['age'].to_numpy())
        self.pyro_model.age_flow_lognorm_loc = age.log().mean().to(self.torch_device).float()
        self.pyro_model.age_flow_lognorm_scale = age.log().std().to(self.torch_device).float()

        duration = torch.from_numpy(self.calabresi_train.csv['duration'].to_numpy())
        self.pyro_model.duration_flow_lognorm_loc = duration.log().mean().to(self.torch_device).float()
        self.pyro_model.duration_flow_lognorm_scale = duration.log().std().to(self.torch_device).float()

        score = torch.from_numpy(self.calabresi_train.csv['score'].to_numpy())
        self.pyro_model.score_flow_lognorm_loc = score.log().mean().to(self.torch_device).float()
        self.pyro_model.score_flow_lognorm_scale = score.log().std().to(self.torch_device).float()

        ventricle_volume = torch.from_numpy(self.calabresi_train.csv['ventricle_volume'].to_numpy())
        self.pyro_model.ventricle_volume_flow_lognorm_loc = ventricle_volume.log().mean().to(self.torch_device).float()
        self.pyro_model.ventricle_volume_flow_lognorm_scale = ventricle_volume.log().std().to(self.torch_device).float()

        brain_volume = torch.from_numpy(self.calabresi_train.csv['brain_volume'].to_numpy())
        self.pyro_model.brain_volume_flow_lognorm_loc = brain_volume.log().mean().to(self.torch_device).float()
        self.pyro_model.brain_volume_flow_lognorm_scale = brain_volume.log().std().to(self.torch_device).float()

        lesion_volume = torch.from_numpy(self.calabresi_train.csv['lesion_volume'].to_numpy())
        self.pyro_model.lesion_volume_flow_lognorm_loc = lesion_volume.log().mean().to(self.torch_device).float()
        self.pyro_model.lesion_volume_flow_lognorm_scale = lesion_volume.log().std().to(self.torch_device).float()

        slice_ventricle_volume = torch.from_numpy(self.calabresi_train.csv['slice_ventricle_volume'].to_numpy())
        self.pyro_model.slice_ventricle_volume_flow_lognorm_loc = slice_ventricle_volume.log().mean().to(self.torch_device).float()
        self.pyro_model.slice_ventricle_volume_flow_lognorm_scale = slice_ventricle_volume.log().std().to(self.torch_device).float()

        slice_brain_volume = torch.from_numpy(self.calabresi_train.csv['slice_brain_volume'].to_numpy())
        self.pyro_model.slice_brain_volume_flow_lognorm_loc = slice_brain_volume.log().mean().to(self.torch_device).float()
        self.pyro_model.slice_brain_volume_flow_lognorm_scale = slice_brain_volume.log().std().to(self.torch_device).float()

        slice_lesion_volume = torch.from_numpy(self.calabresi_train.csv['slice_lesion_volume'].to_numpy())
        self.pyro_model.slice_lesion_volume_flow_lognorm_loc = slice_lesion_volume.log().mean().to(self.torch_device).float()
        self.pyro_model.slice_lesion_volume_flow_lognorm_scale = slice_lesion_volume.log().std().to(self.torch_device).float()

        if self.hparams.validate:
            logger.info(f'set age_flow_lognorm {self.pyro_model.age_flow_lognorm.loc} +/- {self.pyro_model.age_flow_lognorm.scale}')
            logger.info(f'set brain_volume_flow_lognorm {self.pyro_model.brain_volume_flow_lognorm.loc} +/- {self.pyro_model.brain_volume_flow_lognorm.scale}')
            logger.info(f'set ventricle_volume_flow_lognorm {self.pyro_model.ventricle_volume_flow_lognorm.loc} +/- {self.pyro_model.ventricle_volume_flow_lognorm.scale}')  # noqa: E501
            logger.info(f'set lesion_volume_flow_lognorm {self.pyro_model.lesion_volume_flow_lognorm.loc} +/- {self.pyro_model.lesion_volume_flow_lognorm.scale}')  # noqa: E501
            logger.info(f'set slice_brain_volume_flow_lognorm {self.pyro_model.slice_brain_volume_flow_lognorm.loc} +/- {self.pyro_model.slice_brain_volume_flow_lognorm.scale}')
            logger.info(f'set slice_ventricle_volume_flow_lognorm {self.pyro_model.slice_ventricle_volume_flow_lognorm.loc} +/- {self.pyro_model.slice_ventricle_volume_flow_lognorm.scale}')  # noqa: E501
            logger.info(f'set slice_lesion_volume_flow_lognorm {self.pyro_model.slice_lesion_volume_flow_lognorm.loc} +/- {self.pyro_model.slice_lesion_volume_flow_lognorm.scale}')  # noqa: E501
            logger.info(f'set duration_flow_lognorm {self.pyro_model.duration_flow_lognorm.loc} +/- {self.pyro_model.duration_flow_lognorm.scale}')  # noqa: E501
            logger.info(f'set score_flow_lognorm {self.pyro_model.score_flow_lognorm.loc} +/- {self.pyro_model.score_flow_lognorm.scale}')  # noqa: E501

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        num_cpus = len(os.sched_getaffinity(0)) // 2
        on_gpu = torch.cuda.is_available()  # assume if cuda available, we are using it
        return DataLoader(self.calabresi_train, batch_size=self.train_batch_size,
                          shuffle=True, num_workers=num_cpus, pin_memory=on_gpu)

    def val_dataloader(self):
        self.val_loader = DataLoader(self.calabresi_val, batch_size=self.test_batch_size, shuffle=False)
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(self.calabresi_test, batch_size=self.test_batch_size, shuffle=False)
        return self.test_loader

    def forward(self, *args, **kwargs):
        pass

    def prep_batch(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_epoch_end(self, outputs):
        outputs = self.assemble_epoch_end_outputs(outputs)
        metrics = {('val/' + k): v for k, v in outputs.items()}
        if self.current_epoch % self.hparams.sample_img_interval == 0:
            self.sample_images()
        val_loss = metrics['val/loss'] if isinstance(metrics['val/loss'], torch.Tensor) else torch.tensor(metrics['val/loss'])
        return {'val_loss': val_loss, 'log': metrics}

    def test_epoch_end(self, outputs):
        logger.info('Assembling outputs')
        outputs = self.assemble_epoch_end_outputs(outputs)
        samples = outputs.pop('samples')

        sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)
        samples['unconditional_samples'] = {k: sample_trace.nodes['x']['value'].cpu() for k in self.required_data}

        cond_data = {
            'slice_brain_volume': self.slice_brain_volume_range.repeat(self.hparams.test_batch_size, 1),
            'slice_ventricle_volume': self.slice_ventricle_volume_range.repeat(self.hparams.test_batch_size, 1),
            'slice_lesion_volume': self.slice_lesion_volume_range.repeat(self.hparams.test_batch_size, 1),
            'z': torch.randn([self.hparams.test_batch_size, self.hparams.latent_dim], device=self.torch_device, dtype=torch.float).repeat_interleave(9, 0)
        }
        sample_trace = pyro.poutine.trace(pyro.condition(self.pyro_model.sample, data=cond_data)).get_trace(9 * self.hparams.test_batch_size)
        samples['conditional_samples'] = {k: sample_trace.nodes['x']['value'].cpu() for k in self.required_data}

        logger.info(f'Got samples: {tuple(samples.keys())}')
        metrics = {('test/' + k): v for k, v in outputs.items()}
        for k, v in samples.items():
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{k}.pt')
            logging.info(f'Saving samples for {k} to {p}')
            torch.save(v, p)

        p = os.path.join(self.trainer.logger.experiment.log_dir, 'metrics.pt')
        torch.save(metrics, p)
        return {'test_loss': metrics['test/loss'], 'log': metrics}

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
            'do(slice_brain_volume=8000)':  {'slice_brain_volume': torch.ones_like(batch['slice_brain_volume']) * 8000},
            'do(slice_brain_volume=16000)': {'slice_brain_volume': torch.ones_like(batch['slice_brain_volume']) * 16000},
            'do(slice_ventricle_volume=500)':  {'slice_ventricle_volume': torch.ones_like(batch['slice_ventricle_volume']) * 500},
            'do(slice_ventricle_volume=2000)': {'slice_ventricle_volume': torch.ones_like(batch['slice_ventricle_volume']) * 2000},
            'do(slice_lesion_volume=0)':    {'slice_lesion_volume': torch.ones_like(batch['slice_lesion_volume']) * 1e-5},
            'do(slice_lesion_volume=1000)': {'slice_lesion_volume': torch.ones_like(batch['slice_lesion_volume']) * 1000.},
            'do(age=20)': {'age': torch.ones_like(batch['age']) * 20},
            'do(age=60)': {'age': torch.ones_like(batch['age']) * 60},
            'do(sex=0)': {'sex': torch.zeros_like(batch['sex'])},
            'do(sex=1)': {'sex': torch.ones_like(batch['sex'])},
            'do(duration=0)': {'duration': torch.zeros_like(batch['type']) + 1e-5},
            'do(duration=12)': {'duration': torch.ones_like(batch['type']) * 12.},
            'do(score=0)': {'type': torch.ones_like(batch['type']) + 1e-5},
            'do(score=6)': {'type': torch.ones_like(batch['type']) * 6.},
            'do(slice_number=115)': {'type': torch.ones_like(batch['type']) * 115.},
            'do(slice_number=125)': {'type': torch.ones_like(batch['type']) * 125.},
            'do(slice_brain_volume=8000, slice_ventricle_volume=500)': {'slice_brain_volume': torch.ones_like(batch['slice_brain_volume']) * 8000.,
                                                                        'slice_ventricle_volume': torch.ones_like(batch['slice_ventricle_volume']) * 500.},
            'do(slice_brain_volume=16000, slice_ventricle_volume=1000)': {'slice_brain_volume': torch.ones_like(batch['slice_brain_volume']) * 16000.,
                                                                          'slice_ventricle_volume': torch.ones_like(batch['slice_ventricle_volume']) * 1000.}
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
        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            torchvision.utils.save_image(imgs, p)
        grid = torchvision.utils.make_grid(imgs, normalize=normalize, **kwargs)
        self.logger.experiment.add_image(tag, grid, self.current_epoch)

    def get_batch(self, loader):
        batch = next(iter(self.val_loader))
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
                    sns.kdeplot(x=np_val(x), ax=ax[i], fill=True, thresh=0.01)
                elif len(covariates) == 2:
                    (x_n, x), (y_n, y) = tuple(covariates.items())
                    sns.kdeplot(x=np_val(x), y=np_val(y), ax=ax[i], fill=True, thresh=0.01)
                    ax[i].set_ylabel(y_n)
                else:
                    raise ValueError(f'got too many values: {len(covariates)}')
            except np.linalg.LinAlgError:
                logging.info(f'got a linalg error when plotting {tag}/{name}')
                raise

            ax[i].set_title(name)
            ax[i].set_xlabel(x_n)

        sns.despine()

        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            plt.savefig(p, dpi=300)

        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

    def build_reconstruction(self, obs, tag='reconstruction'):
        self._check_observation(obs)
        x = obs['x']
        recon = self.pyro_model.reconstruct(obs, num_particles=self.hparams.num_sample_particles)
        self.log_img_grid(tag, torch.cat([x, recon], 0))
        self.logger.experiment.add_scalar(f'{tag}/mse', torch.mean(torch.square(x - recon).sum((1, 2, 3))), self.current_epoch)

    @property
    def required_data(self):
        return {'x', 'sex', 'age', 'ventricle_volume', 'brain_volume', 'lesion_volume',
                'slice_ventricle_volume', 'slice_brain_volume', 'slice_lesion_volume',
                'score', 'duration', 'slice_number'}

    def _check_observation(self, obs):
        keys = obs.keys()
        assert self.required_data.issubset(set(keys)), f'Incompatible observation: {tuple(keys)}'

    def build_counterfactual(self, tag, obs, conditions, absolute=None):
        self._check_observation(obs)
        imgs = [obs['x']]
        if absolute == 'slice_brain_volume':
            sampled_kdes = {'orig': {'slice_brain_volume': obs['slice_brain_volume']}}
        elif absolute == 'ventricle_volume':
            sampled_kdes = {'orig': {'slice_ventricle_volume': obs['slice_ventricle_volume']}}
        elif absolute == 'lesion_volume':
            sampled_kdes = {'orig': {'slice_lesion_volume': obs['slice_lesion_volume']}}
        elif absolute == 'score':
            sampled_kdes = {'orig': {'score': obs['score']}}
        elif absolute == 'duration':
            sampled_kdes = {'orig': {'duration': obs['duration']}}
        else:
            sampled_kdes = {'orig': {'slice_brain_volume': obs['slice_brain_volume'],
                                     'slice_ventricle_volume': obs['slice_ventricle_volume']}}

        for name, data in conditions.items():
            counterfactual = self.pyro_model._gen_counterfactual(obs=obs, condition=data)

            counter = counterfactual['x']
            sampled_brain_volume = counterfactual['slice_brain_volume']
            sampled_ventricle_volume = counterfactual['slice_ventricle_volume']
            sampled_lesion_volume = counterfactual['slice_lesion_volume']
            sampled_score = counterfactual['score']
            sampled_duration = counterfactual['duration']

            imgs.append(counter)
            if absolute == 'slice_brain_volume':
                sampled_kdes[name] = {'slice_brain_volume': sampled_brain_volume}
            elif absolute == 'slice_ventricle_volume':
                sampled_kdes[name] = {'slice_ventricle_volume': sampled_ventricle_volume}
            elif absolute == 'slice_lesion_volume':
                sampled_kdes[name] = {'slice_lesion_volume': sampled_lesion_volume}
            elif absolute == 'score':
                sampled_kdes[name] = {'score': sampled_score}
            elif absolute == 'duration':
                sampled_kdes[name] = {'duration': sampled_duration}
            else:
                sampled_kdes[name] = {'slice_brain_volume': sampled_brain_volume,
                                      'slice_ventricle_volume': sampled_ventricle_volume}

        self.log_img_grid(tag, torch.cat(imgs, 0))
        self.log_kdes(f'{tag}_sampled', sampled_kdes, save_img=True)

    def sample_images(self):
        with torch.no_grad():
            sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)

            samples = sample_trace.nodes['x']['value']
            sampled_brain_volume = sample_trace.nodes['slice_brain_volume']['value']
            sampled_ventricle_volume = sample_trace.nodes['slice_ventricle_volume']['value']

            s = samples.shape[0] // 8
            self.log_img_grid('samples', samples.data[::s])

            cond_data = {'slice_brain_volume': self.slice_brain_volume_range,
                         'slice_ventricle_volume': self.slice_ventricle_volume_range,
                         'slice_lesion_volume': self.slice_lesion_volume_range,
                         'z': self.z_range}
            samples = pyro.condition(self.pyro_model.sample, data=cond_data)(9)['x']
            self.log_img_grid('cond_samples', samples.data, nrow=3)

            obs_batch = self.prep_batch(self.get_batch(self.val_loader))

            kde_data = {
                'batch': {'slice_brain_volume': obs_batch['slice_brain_volume'],
                          'slice_ventricle_volume': obs_batch['slice_ventricle_volume']},
                'sampled': {'slice_brain_volume': sampled_brain_volume,
                            'slice_ventricle_volume': sampled_ventricle_volume}
            }
            self.log_kdes('sample_kde', kde_data, save_img=True)

            exogeneous = self.pyro_model.infer(obs_batch)

            for (tag, val) in exogeneous.items():
                self.logger.experiment.add_histogram(tag, val, self.current_epoch)

            obs_batch = {k: v[::s] for k, v in obs_batch.items()}

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
                '10000': {'slice_brain_volume': torch.zeros_like(obs_batch['slice_brain_volume']) + 10000},
                '16000': {'slice_brain_volume': torch.zeros_like(obs_batch['slice_brain_volume']) + 16000}
            }
            self.build_counterfactual('do(slice_brain_volume=x)', obs=obs_batch, conditions=conditions, absolute='slice_brain_volume')

            conditions = {
                '10':   {'slice_ventricle_volume': torch.zeros_like(obs_batch['slice_ventricle_volume']) + 10},
                '1000': {'slice_ventricle_volume': torch.zeros_like(obs_batch['slice_ventricle_volume']) + 1000},
                '2000': {'slice_ventricle_volume': torch.zeros_like(obs_batch['slice_ventricle_volume']) + 2000},

            }
            self.build_counterfactual('do(slice_ventricle_volume=x)', obs=obs_batch, conditions=conditions, absolute='slice_ventricle_volume')

            conditions = {
                '0':    {'slice_lesion_volume': torch.zeros_like(obs_batch['slice_lesion_volume']) + 1e-5},
                '500':  {'slice_lesion_volume': torch.zeros_like(obs_batch['slice_lesion_volume']) + 500},
                '1000': {'slice_lesion_volume': torch.zeros_like(obs_batch['slice_lesion_volume']) + 1000},

            }
            self.build_counterfactual('do(slice_lesion_volume=x)', obs=obs_batch, conditions=conditions, absolute='slice_lesion_volume')

            conditions = {
                '1': {'score': torch.zeros_like(obs_batch['score']) + 1.},
                '5': {'score': torch.zeros_like(obs_batch['score']) + 5.}
            }
            self.build_counterfactual('do(score=x)', obs=obs_batch, conditions=conditions, absolute='score')

            conditions = {
                '0': {'duration': torch.zeros_like(obs_batch['duration']) + 1e-5},
                '10': {'duration': torch.zeros_like(obs_batch['duration']) + 10.},
            }
            self.build_counterfactual('do(duration=x)', obs=obs_batch, conditions=conditions, absolute='duration')

            conditions = {
                '115': {'slice_number': torch.zeros_like(obs_batch['slice_number']) + 115},
                '125': {'slice_number': torch.zeros_like(obs_batch['slice_number']) + 125.}
            }
            self.build_counterfactual('do(slice_number=x)', obs=obs_batch, conditions=conditions, absolute='slice_number')

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--train-csv', default="/iacl/pg20/jacobr/calabresi/png/csv/train_png.csv", type=str, help="csv for training data (default: %(default)s)")  # noqa: E501
        parser.add_argument('--valid-csv', default="/iacl/pg20/jacobr/calabresi/png/csv/valid_png.csv", type=str, help="csv for validation data (default: %(default)s)")  # noqa: E501
        parser.add_argument('--test-csv', default="/iacl/pg20/jacobr/calabresi/png/csv/test_png.csv", type=str, help="csv for testing data (default: %(default)s)")  # noqa: E501
        parser.add_argument('--crop-size', default=(192,192), type=int, nargs=2, help="size of patch to take from image (default: %(default)s)")
        parser.add_argument('--sample-img-interval', default=10, type=int, help="interval in which to sample and log images (default: %(default)s)")
        parser.add_argument('--train-batch-size', default=256, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--test-batch-size', default=64, type=int, help="test batch size (default: %(default)s)")
        parser.add_argument('--validate', default=False, action='store_true', help="whether to validate (default: %(default)s)")
        parser.add_argument('--lr', default=1e-4, type=float, help="lr of deep part (default: %(default)s)")
        parser.add_argument('--pgm-lr', default=5e-3, type=float, help="lr of pgm (default: %(default)s)")
        parser.add_argument('--l2', default=0., type=float, help="weight decay (default: %(default)s)")
        parser.add_argument('--use-amsgrad', default=False, action='store_true', help="use amsgrad? (default: %(default)s)")
        parser.add_argument('--train-crop-type', default='random', choices=['random', 'center'], help="how to crop training images (default: %(default)s)")
        return parser
