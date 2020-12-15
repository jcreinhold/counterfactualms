from collections import defaultdict
import logging
from typing import Mapping, Tuple

import numpy as np
import pyro
from pyro.infer import SVI, TraceGraph_ELBO, Trace_ELBO
from pyro.nn import pyro_method
from pyro.optim import ExponentialLR, AdagradRMSProp  # noqa: F401
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.transforms import (
    ComposeTransform, AffineTransform, ExpTransform, Spline, Permute
)
from pyro.distributions.transforms import batchnorm, iterated
from pyro.distributions import (
    LowRankMultivariateNormal, MultivariateNormal, Normal, Laplace, TransformedDistribution  # noqa: F401
)
import torch
from torch.distributions import Independent
from torch.optim import AdamW

from counterfactualms.arch.medical import Decoder, Encoder
from counterfactualms.arch.nvae import Decoder as NDecoder
from counterfactualms.arch.nvae import Encoder as NEncoder
from counterfactualms.arch.thirdparty.neural_operations import Swish
from counterfactualms.distributions.transforms.reshape import ReshapeTransform
from counterfactualms.distributions.transforms.affine import LowerCholeskyAffine
from counterfactualms.pyro_modifications import spline_autoregressive, spline_coupling
from counterfactualms.distributions.deep import (
    DeepMultivariateNormal, DeepIndepNormal, DeepIndepMixtureNormal, Conv2dIndepNormal, DeepLowRankMultivariateNormal
)
from counterfactualms.experiments.calabresi.base_experiment import (
    BaseCovariateExperiment, BaseSEM, EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401
)

logger = logging.getLogger(__name__)


class StorageTraceGraph_ELBO(TraceGraph_ELBO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)
        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = guide_trace
        return model_trace, guide_trace


class StorageTrace_ELBO(Trace_ELBO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)
        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = guide_trace
        return model_trace, guide_trace


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class BaseVISEM(BaseSEM):
    context_dim = 0  # number of context dimensions for decoder

    def __init__(self, latent_dim:int, prior_components:int=1, posterior_components:int=1,
                 logstd_init:float=-5, enc_filters:Tuple[int]=(16,32,64,128),
                 dec_filters:Tuple[int]=(128,64,32,16), num_convolutions:int=3, use_upconv:bool=False,
                 decoder_type:str='fixed_var', decoder_cov_rank:int=10, img_shape:Tuple[int]=(128,128),
                 use_nvae=False, use_weight_norm=False, use_spectral_norm=False, laplace_likelihood=False,
                 eps=0.1, n_prior_flows=3, n_posterior_flows=3, use_autoregressive=False, use_swish=False, **kwargs):
        super().__init__(**kwargs)
        self.img_shape = (1,) + tuple(img_shape)
        self.latent_dim = latent_dim
        self.prior_components = prior_components
        self.posterior_components = posterior_components
        self.logstd_init = logstd_init
        self.enc_filters = enc_filters
        self.dec_filters = dec_filters
        self.num_convolutions = num_convolutions
        self.use_upconv = use_upconv
        self.decoder_type = decoder_type
        self.decoder_cov_rank = decoder_cov_rank
        self.use_nvae = use_nvae
        self.use_weight_norm = use_weight_norm
        self.use_spectral_norm = use_spectral_norm
        self.laplace_likelihood = laplace_likelihood
        self.eps = eps
        self.n_prior_flows = n_prior_flows
        self.n_posterior_flows = n_posterior_flows
        self.use_autoregressive = use_autoregressive
        self.use_swish = use_swish
        self.annealing_factor = 1.  # initialize here; will be changed during training

        # decoder parts
        if use_nvae:
            decoder = NDecoder(
                num_convolutions=self.num_convolutions, filters=self.dec_filters,
                latent_dim=self.latent_dim + self.context_dim,
                output_size=self.img_shape
            )
        else:
            decoder = Decoder(
                num_convolutions=self.num_convolutions, filters=self.dec_filters,
                latent_dim=self.latent_dim + self.context_dim, upconv=self.use_upconv,
                output_size=self.img_shape,
                use_weight_norm=self.use_weight_norm,
                use_spectral_norm=self.use_spectral_norm
            )

        if self.decoder_type == 'fixed_var':
            self.decoder = Conv2dIndepNormal(decoder, 1, 1)
            torch.nn.init.zeros_(self.decoder.logvar_head.weight)
            self.decoder.logvar_head.weight.requires_grad = False
            torch.nn.init.constant_(self.decoder.logvar_head.bias, self.logstd_init)
            self.decoder.logvar_head.bias.requires_grad = False

        elif self.decoder_type == 'learned_var':
            self.decoder = Conv2dIndepNormal(decoder, 1, 1)
            torch.nn.init.zeros_(self.decoder.logvar_head.weight)
            self.decoder.logvar_head.weight.requires_grad = False
            torch.nn.init.constant_(self.decoder.logvar_head.bias, self.logstd_init)
            self.decoder.logvar_head.bias.requires_grad = True

        elif self.decoder_type == 'independent_var':
            self.decoder = Conv2dIndepNormal(decoder, 1, 1)
            torch.nn.init.zeros_(self.decoder.logvar_head.weight)
            self.decoder.logvar_head.weight.requires_grad = True
            torch.nn.init.normal_(self.decoder.logvar_head.bias, self.logstd_init, 1e-1)
            self.decoder.logvar_head.bias.requires_grad = True

        elif self.decoder_type == 'multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepMultivariateNormal(seq, np.prod(self.img_shape), np.prod(self.img_shape))

        elif self.decoder_type == 'sharedvar_multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepMultivariateNormal(seq, np.prod(self.img_shape), np.prod(self.img_shape))
            torch.nn.init.zeros_(self.decoder.logdiag_head.weight)
            self.decoder.logdiag_head.weight.requires_grad = False
            torch.nn.init.zeros_(self.decoder.lower_head.weight)
            self.decoder.lower_head.weight.requires_grad = False
            torch.nn.init.normal_(self.decoder.logdiag_head.bias, self.logstd_init, 1e-1)
            self.decoder.logdiag_head.bias.requires_grad = True

        elif self.decoder_type == 'lowrank_multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepLowRankMultivariateNormal(
                seq, np.prod(self.img_shape), np.prod(self.img_shape), decoder_cov_rank
            )

        elif self.decoder_type == 'sharedvar_lowrank_multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepLowRankMultivariateNormal(
                seq, np.prod(self.img_shape), np.prod(self.img_shape), decoder_cov_rank
            )
            torch.nn.init.zeros_(self.decoder.logdiag_head.weight)
            self.decoder.logdiag_head.weight.requires_grad = False
            torch.nn.init.zeros_(self.decoder.factor_head.weight)
            self.decoder.factor_head.weight.requires_grad = False
            torch.nn.init.normal_(self.decoder.logdiag_head.bias, self.logstd_init, 1e-1)
            self.decoder.logdiag_head.bias.requires_grad = True

        else:
            raise ValueError(f'unknown decoder type {self.decoder_type}.')

        # encoder parts
        if self.use_nvae:
            self.encoder = NEncoder(
                num_convolutions=self.num_convolutions,
                filters=self.enc_filters,
                latent_dim=self.latent_dim,
                input_size=self.img_shape
            )
        else:
            self.encoder = Encoder(
                num_convolutions=self.num_convolutions,
                filters=self.enc_filters,
                latent_dim=self.latent_dim,
                input_size=self.img_shape,
                use_weight_norm=self.use_weight_norm,
                use_spectral_norm = self.use_spectral_norm
            )

        nonlinearity = Swish() if self.use_swish else torch.nn.LeakyReLU(0.1)
        latent_layers = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim + self.context_dim, self.latent_dim),
            nonlinearity
        )

        if self.posterior_components > 1:
            self.latent_encoder = DeepIndepMixtureNormal(
                latent_layers, self.latent_dim, self.latent_dim, self.posterior_components)
        else:
            self.latent_encoder = DeepIndepNormal(latent_layers, self.latent_dim, self.latent_dim)

        if self.prior_components > 1:
            self.z_loc = torch.nn.Parameter(torch.randn([self.prior_components, self.latent_dim]))
            self.z_scale = torch.nn.Parameter(torch.randn([self.latent_dim]).clamp(min=-1.,max=None))  # log scale
            self.register_buffer('z_components',  # don't be bayesian about the mixture components
                ((1/self.prior_components)*torch.ones([self.prior_components], requires_grad=False)).log())
        else:
            self.register_buffer('z_loc', torch.zeros([latent_dim, ], requires_grad=False))
            self.register_buffer('z_scale', torch.ones([latent_dim, ], requires_grad=False))
            self.z_components = None

        # priors
        self.sex_logits = torch.nn.Parameter(torch.zeros([1, ]))

        for k in self.required_data - {'sex', 'x'}:
            self.register_buffer(f'{k}_base_loc', torch.zeros([1, ], requires_grad=False))
            self.register_buffer(f'{k}_base_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('x_base_loc', torch.zeros(self.img_shape, requires_grad=False))
        self.register_buffer('x_base_scale', torch.ones(self.img_shape, requires_grad=False))

        for k in self.required_data - {'sex', 'x'}:
            self.register_buffer(f'{k}_flow_lognorm_loc', torch.zeros([], requires_grad=False))
            self.register_buffer(f'{k}_flow_lognorm_scale', torch.ones([], requires_grad=False))

        perm = lambda: torch.randperm(self.latent_dim, dtype=torch.long, requires_grad=False)
        for i in range(self.n_prior_flows):
            self.register_buffer(f'prior_flow_permutation_{i}', perm())
        for i in range(self.n_posterior_flows):
            self.register_buffer(f'posterior_flow_permutation_{i}', perm())

        # age flow
        self.age_flow_components = ComposeTransformModule([Spline(1)])
        self.age_flow_lognorm = AffineTransform(loc=self.age_flow_lognorm_loc.item(), scale=self.age_flow_lognorm_scale.item())
        self.age_flow_constraint_transforms = ComposeTransform([self.age_flow_lognorm, ExpTransform()])
        self.age_flow_transforms = ComposeTransform([self.age_flow_components, self.age_flow_constraint_transforms])

        # other flows shared components
        self.ventricle_volume_flow_lognorm = AffineTransform(loc=self.ventricle_volume_flow_lognorm_loc.item(), scale=self.ventricle_volume_flow_lognorm_scale.item())  # noqa: E501
        self.ventricle_volume_flow_constraint_transforms = ComposeTransform([self.ventricle_volume_flow_lognorm, ExpTransform()])

        self.brain_volume_flow_lognorm = AffineTransform(loc=self.brain_volume_flow_lognorm_loc.item(), scale=self.brain_volume_flow_lognorm_scale.item())
        self.brain_volume_flow_constraint_transforms = ComposeTransform([self.brain_volume_flow_lognorm, ExpTransform()])

        self.lesion_volume_flow_lognorm = AffineTransform(loc=self.lesion_volume_flow_lognorm_loc.item(), scale=self.lesion_volume_flow_lognorm_scale.item())
        self.lesion_volume_flow_eps = AffineTransform(loc=-eps, scale=1.)
        self.lesion_volume_flow_constraint_transforms = ComposeTransform([self.lesion_volume_flow_lognorm, ExpTransform(), self.lesion_volume_flow_eps])

        self.duration_flow_lognorm = AffineTransform(loc=self.duration_flow_lognorm_loc.item(), scale=self.duration_flow_lognorm_scale.item())
        self.duration_flow_eps = AffineTransform(loc=-eps, scale=1.)
        self.duration_flow_constraint_transforms = ComposeTransform([self.duration_flow_lognorm, ExpTransform(), self.duration_flow_eps])

        self.score_flow_lognorm = AffineTransform(loc=self.score_flow_lognorm_loc.item(), scale=self.score_flow_lognorm_scale.item())
        self.score_flow_eps = AffineTransform(loc=-eps, scale=1.)
        self.score_flow_constraint_transforms = ComposeTransform([self.score_flow_lognorm, ExpTransform(), self.score_flow_eps])

        spline_kwargs = dict(hidden_dims=(2*self.latent_dim, 2*self.latent_dim), nonlinearity=nonlinearity)
        spline_ = spline_autoregressive if self.use_autoregressive else spline_coupling
        self.use_prior_flow = self.n_prior_flows > 0
        self.prior_affine = iterated(self.n_prior_flows, batchnorm, self.latent_dim, momentum=0.05) if self.use_prior_flow else []
        self.prior_permutations = [Permute(getattr(self, f'prior_flow_permutation_{i}')) for i in range(self.n_prior_flows)]
        self.prior_flow_components = iterated(self.n_prior_flows, spline_, self.latent_dim, **spline_kwargs) if self.use_prior_flow else []
        self.prior_flow_transforms = [
            x for c in zip(self.prior_permutations, self.prior_affine, self.prior_flow_components) for x in c
        ]

        self.use_posterior_flow = self.n_posterior_flows > 0
        self.posterior_affine = iterated(self.n_posterior_flows, batchnorm, self.latent_dim, momentum=0.05) if self.use_posterior_flow else []
        self.posterior_permutations = [Permute(getattr(self, f'posterior_flow_permutation_{i}')) for i in range(self.n_posterior_flows)]
        self.posterior_flow_components = iterated(self.n_posterior_flows, spline_, self.latent_dim, **spline_kwargs) if self.use_posterior_flow else []
        self.posterior_flow_transforms = [
            x for c in zip(self.posterior_permutations, self.posterior_affine, self.posterior_flow_components) for x in c
        ]

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if 'flow_lognorm_loc' in name:
            name_ = name.replace('flow_lognorm_loc', '')
            getattr(self, name_ + 'flow_lognorm').loc = value.item()
        elif 'flow_lognorm_scale' in name:
            name_ = name.replace('flow_lognorm_scale', '')
            getattr(self, name_ + 'flow_lognorm').scale = value.item()
        elif 'flow_norm_loc' in name:
            name_ = name.replace('flow_norm_loc', '')
            getattr(self, name_ + 'flow_norm').loc = value.item()
        elif 'flow_norm_scale' in name:
            name_ = name.replace('flow_norm_scale', '')
            getattr(self, name_ + 'flow_norm').scale = value.item()
        elif 'prior_flow_permutation' in name:
            i = int(name[-1])
            self.prior_permutations[i].permutation = value
        elif 'posterior_flow_permutation' in name:
            i = int(name[-1])
            self.posterior_permutations[i].permutation = value

    def _get_preprocess_transforms(self):
        return super()._get_preprocess_transforms().inv

    def _get_transformed_x_dist(self, latent):
        x_pred_dist = self.decoder.predict(latent)  # returns a normal dist with mean of the predicted image
        if self.laplace_likelihood:
            x_base_dist = Laplace(self.x_base_loc, self.x_base_scale).to_event(3)
        else:
            x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3)  # 3 dimensions starting from right dep.

        preprocess_transform = self._get_preprocess_transforms()

        if isinstance(x_pred_dist, MultivariateNormal) or isinstance(x_pred_dist, LowRankMultivariateNormal):
            chol_transform = LowerCholeskyAffine(x_pred_dist.loc, x_pred_dist.scale_tril)
            reshape_transform = ReshapeTransform(self.img_shape, (np.prod(self.img_shape), ))
            x_reparam_transform = ComposeTransform([reshape_transform, chol_transform, reshape_transform.inv])
        elif isinstance(x_pred_dist, Independent):
            x_pred_dist = x_pred_dist.base_dist
            x_reparam_transform = AffineTransform(x_pred_dist.loc, x_pred_dist.scale, 3)
        else:
            raise ValueError(f'{x_pred_dist} not valid.')

        return TransformedDistribution(x_base_dist, ComposeTransform([x_reparam_transform, preprocess_transform]))

    @pyro_method
    def guide(self, obs):
        raise NotImplementedError()

    @pyro_method
    def svi_guide(self, obs):
        self._check_observation(obs)
        self.guide(obs)

    @pyro_method
    def svi_model(self, obs):
        self._check_observation(obs)
        batch_size = obs['x'].shape[0]
        with pyro.plate('observations', batch_size):
            pyro.condition(self.model, data=obs)()

    @pyro_method
    def infer_z(self, *args, **kwargs):
        return self.guide(*args, **kwargs)

    @property
    def required_data(self):
        return {'x', 'sex', 'age', 'ventricle_volume', 'brain_volume', 'lesion_volume',
                'score', 'duration'}

    def _check_observation(self, obs):
        keys = obs.keys()
        assert self.required_data == set(keys), f'Incompatible observation: {tuple(keys)}'

    @pyro_method
    def infer(self, obs):
        self._check_observation(obs)
        obs_ = obs.copy()
        z = self.infer_z(obs_)
        obs_.update(dict(z=z))
        exogenous = self.infer_exogenous(obs_)
        exogenous['z'] = z
        return exogenous

    @pyro_method
    def reconstruct(self, obs, num_particles:int=1):
        self._check_observation(obs)
        z_dist = pyro.poutine.trace(self.guide).get_trace(obs).nodes['z']['fn']
        batch_size = obs['x'].shape[0]
        obs_ = {k: v for k, v in obs.items() if k != 'x'}
        recons = []
        for _ in range(num_particles):
            z = pyro.sample('z', z_dist)
            obs_.update({'z': z})
            recon = pyro.poutine.condition(
                self.sample, data=obs_)(batch_size)
            recons += [recon['x']]
        return torch.stack(recons).mean(0)

    def _cf_dict(self, counterfactuals):
        out = {k: [] for k in self.required_data}
        for cf in counterfactuals:
            for k in self.required_data:
                out[k].append(cf[k])
        out = {k: torch.stack(v).mean(0) for k, v in out.items()}
        return out

    @pyro_method
    def counterfactual(self, obs, condition:Mapping=None, num_particles:int=1):
        self._check_observation(obs)
        obs_ = obs.copy()
        z_dist = pyro.poutine.trace(self.guide).get_trace(obs_).nodes['z']['fn']  # variational posterior
        n = obs_['x'].shape[0]

        counterfactuals = []
        for _ in range(num_particles):
            z = pyro.sample('z', z_dist)
            obs_.update(dict(z=z))
            exogenous = self.infer_exogenous(obs_)
            exogenous['z'] = z
            # condition on these vars if they aren't included in 'do' as they are root nodes
            # and we don't have the exogenous noise for them yet
            if 'sex' not in condition.keys():
                exogenous['sex'] = obs_['sex']

            cf = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogenous), data=condition)(n)
            counterfactuals += [cf]

        return self._cf_dict(counterfactuals)

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)
        parser.add_argument('--latent-dim', default=100, type=int, help="latent dimension of model (default: %(default)s)")
        parser.add_argument('--prior-components', default=1, type=int, help="number of mixture components for prior (default: %(default)s)")
        parser.add_argument('--posterior-components', default=1, type=int, help="number of mixture components for posterior (default: %(default)s)")
        parser.add_argument('--logstd-init', default=-5, type=float, help="init of logstd (default: %(default)s)")
        parser.add_argument('--enc-filters', default=[16,32,64,128,256], nargs='+', type=int, help="number of filters in each layer of encoder (default: %(default)s)")
        parser.add_argument('--dec-filters', default=[256,128,64,32,16], nargs='+', type=int, help="number of filters in each layer of decoder (default: %(default)s)")
        parser.add_argument('--num-convolutions', default=3, type=int, help="number of convolutions in each layer (default: %(default)s)")
        parser.add_argument('--use-upconv', default=False, action='store_true', help="use upsample->conv instead of transpose conv (default: %(default)s)")
        parser.add_argument('--use-nvae', default=False, action='store_true', help="use nvae instead of standard vae (default: %(default)s)")
        parser.add_argument('--use-weight-norm', default=False, action='store_true', help="use weight norm in conv layers (not w/ nvae) (default: %(default)s)")
        parser.add_argument('--use-spectral-norm', default=False, action='store_true', help="use spectral norm in conv layers (not w/ nvae) (default: %(default)s)")
        parser.add_argument('--laplace-likelihood', default=False, action='store_true', help="use laplace likelihood for image (default: %(default)s)")
        parser.add_argument('--n-prior-flows', default=3, type=int, help="use this number of flows for prior in flow net (default: %(default)s)")
        parser.add_argument('--n-posterior-flows', default=3, type=int, help="use this number of flows for posterior in flow net (default: %(default)s)")
        parser.add_argument('--use-autoregressive', default=False, action='store_true', help="use autoregressive spline for prior/post instead of coupling (default: %(default)s)")
        parser.add_argument('--use-swish', default=False, action='store_true', help="use swish in spline for nonlinearity (default: %(default)s)")
        parser.add_argument(
            '--decoder-type', default='fixed_var', help="var type (default: %(default)s)",
            choices=['fixed_var', 'learned_var', 'independent_var', 'sharedvar_multivariate_gaussian',
                     'multivariate_gaussian', 'sharedvar_lowrank_multivariate_gaussian', 'lowrank_multivariate_gaussian'])
        parser.add_argument('--decoder-cov-rank', default=10, type=int, help="rank for lowrank cov approximation (requires lowrank decoder) (default: %(default)s)")  # noqa: E501
        return parser


class SVIExperiment(BaseCovariateExperiment):
    def __init__(self, hparams, pyro_model: BaseSEM):
        super().__init__(hparams, pyro_model)
        if hparams.tracegraph_elbo:
            self.svi_loss = StorageTraceGraph_ELBO(num_particles=hparams.num_svi_particles)
        else:
            self.svi_loss = StorageTrace_ELBO(num_particles=hparams.num_svi_particles)
        self._build_svi()

    def _build_svi(self, loss=None):
        def per_param_callable(module_name, param_name):
            if self.hparams.use_adagrad_rmsprop:
                params = {'eta': self.hparams.eta, 'delta': self.hparams.delta, 't': self.hparams.t}
            else:
                params = {'weight_decay': self.hparams.weight_decay,
                          'betas': self.hparams.betas, 'eps': 1e-5}
                if any([(pn in module_name) for pn in ('prior_flow', 'posterior_flow')]):
                    params['lr'] = self.hparams.lr
                elif 'affine' in module_name:
                    params['lr'] = self.hparams.lr
                    params['weight_decay'] = 0.
                elif 'flow_components' in module_name:
                    params['lr'] = self.hparams.pgm_lr
                elif 'sex_logits' in param_name:
                    params['lr'] = self.hparams.pgm_lr
                    params['weight_decay'] = 0.
                else:
                    params['lr'] = self.hparams.lr
                logger.info(f'building opt for {module_name} - {param_name} with p: {params}')
            return params

        def per_param_clip_args(module_name, param_name):
            clip_args = defaultdict(lambda: None)
            if any([(pn in param_name) for pn in ('affine', 'sex_logits', 'flow_components')]):
                clip_args['clip_norm'] = self.hparams.pgm_clip_norm
            else:
                clip_args['clip_norm'] = self.hparams.clip_norm
            logger.info(f'building clip args for {module_name} - {param_name} with p: {clip_args}')
            return clip_args

        if loss is None:
            loss = self.svi_loss

        optimizer = AdagradRMSProp if self.hparams.use_adagrad_rmsprop else AdamW
        scheduler = ExponentialLR({'optimizer': optimizer, 'optim_args': per_param_callable,
                                   'gamma': self.hparams.lrd}, clip_args=per_param_clip_args)
        if self.hparams.use_cf_guide:
            def guide(*args, **kwargs):
                return self.pyro_model.counterfactual_guide(*args, **kwargs, counterfactual_type=self.hparams.cf_elbo_type)
            self.svi = SVI(self.pyro_model.svi_model, guide, scheduler, loss)
        else:
            self.svi = SVI(self.pyro_model.svi_model, self.pyro_model.svi_guide, scheduler, loss)
        self.svi.loss_class = loss

    def backward(self, *args, **kwargs):
        pass  # No loss to backpropagate since we're using Pyro's optimisation machinery

    def print_trace_updates(self, batch):
        with torch.no_grad():
            logger.info('Traces:\n' + ('#' * 10))

            guide_trace = pyro.poutine.trace(self.pyro_model.svi_guide).get_trace(batch)
            model_trace = pyro.poutine.trace(pyro.poutine.replay(self.pyro_model.svi_model, trace=guide_trace)).get_trace(batch)

            guide_trace = pyro.poutine.util.prune_subsample_sites(guide_trace)
            model_trace = pyro.poutine.util.prune_subsample_sites(model_trace)

            model_trace.compute_log_prob()
            guide_trace.compute_score_parts()

            logging.info(f'model: {model_trace.nodes.keys()}')
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    fn = site['fn']
                    if isinstance(fn, Independent):
                        fn = fn.base_dist
                    try:
                        logging.info(f'{name}: {fn} - {fn.support}')
                    except NotImplementedError:
                        logging.info(f'{name}: {fn}')
                    log_prob_sum = site["log_prob_sum"]
                    is_obs = site["is_observed"]
                    logging.info(f'model - log p({name}) = {log_prob_sum} | obs={is_obs}')
                    if torch.isnan(log_prob_sum):
                        value = site['value'][0]
                        conc0 = fn.concentration0
                        conc1 = fn.concentration1
                        raise RuntimeError(f'Error: \n{value}\n{conc0}\n{conc1}')

            logging.info(f'guide: {guide_trace.nodes.keys()}')

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    fn = site['fn']
                    if isinstance(fn, Independent):
                        fn = fn.base_dist
                    try:
                        logging.info(f'{name}: {fn} - {fn.support}')
                    except NotImplementedError:
                        logging.info(f'{name}: {fn}')
                    entropy = site["score_parts"].entropy_term.sum()
                    is_obs = site["is_observed"]
                    logging.info(f'guide - log q({name}) = {entropy} | obs={is_obs}')

    def get_trace_metrics(self, batch):
        metrics = {}
        model = self.svi.loss_class.trace_storage['model']
        guide = self.svi.loss_class.trace_storage['guide']
        for k in self.required_data:
            metrics[f'log p({k})'] = model.nodes[k]['log_prob'].mean()
        metrics['log p(z)'] = model.nodes['z']['log_prob'].mean()
        metrics['log q(z)'] = guide.nodes['z']['log_prob'].mean()
        metrics['log p(z) - log q(z)'] = metrics['log p(z)'] - metrics['log q(z)']
        return metrics

    def prep_batch(self, batch):
        x = 255. * batch['image'].float()  # multiply by 255 b/c preprocess tfms
        x += (torch.rand_like(x) - 0.5) # add noise per Theis 2016
        out = dict(x=x)
        for k in self.required_data:
            if k in batch:
                out[k] = batch[k].unsqueeze(1).float()
        return out

    def _set_annealing_factor(self, batch_idx=None):
        n_batches_per_epoch = len(self.calabresi_train) // self.train_batch_size
        if batch_idx is None:
            batch_idx = n_batches_per_epoch
        not_in_sanity_check = self.hparams.annealing_epochs > 0
        in_annealing_epochs = self.current_epoch < self.hparams.annealing_epochs
        if not_in_sanity_check and in_annealing_epochs and self.training:
            min_af = self.hparams.min_annealing_factor
            max_af = self.hparams.max_annealing_factor
            self.pyro_model.annealing_factor = min_af + (max_af - min_af) * \
                               (float(batch_idx + self.current_epoch * n_batches_per_epoch + 1) /
                                float(self.hparams.annealing_epochs * n_batches_per_epoch))
        else:
            self.pyro_model.annealing_factor = self.hparams.max_annealing_factor

    def training_step(self, batch, batch_idx):
        self._set_annealing_factor(batch_idx)
        batch = self.prep_batch(batch)
        if self.hparams.validate:
            logging.info('Validation:')
            self.print_trace_updates(batch)
        loss = self.svi.step(batch)
        loss = torch.as_tensor(loss)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        af = self.pyro_model.annealing_factor
        self.log('annealing_factor', af, on_step=False, on_epoch=True)
        metrics = self.get_trace_metrics(batch)
        if np.isnan(loss):
            self.logger.experiment.add_text('nan', f'nand at {self.current_epoch}:\n{metrics}')
            raise ValueError('loss went to nan with metrics:\n{}'.format(metrics))
        for k, v in metrics.items():
            self.log('train/' + k, v, on_step=False, on_epoch=True)
        self.log('klz', metrics['log p(z) - log q(z)'], on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._set_annealing_factor()
        batch = self.prep_batch(batch)
        loss = self.svi.evaluate_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        metrics = self.get_trace_metrics(batch)
        for k, v in metrics.items():
            self.log('val/' + k, v, on_step=False, on_epoch=True)
        return metrics

    def test_step(self, batch, batch_idx):
        self._set_annealing_factor()
        batch = self.prep_batch(batch)
        loss = self.svi.evaluate_loss(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        metrics = self.get_trace_metrics(batch)
        for k, v in metrics.items():
            self.log('test/' + k, v, on_step=False, on_epoch=True)
        samples = self.build_test_samples(batch)
        return {'samples': samples, 'metrics': metrics}

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)
        parser.add_argument('--num-svi-particles', default=4, type=int, help="number of particles to use for ELBO (default: %(default)s)")
        parser.add_argument('--num-sample-particles', default=32, type=int, help="number of particles to use for MC sampling (default: %(default)s)")
        parser.add_argument('--use-cf-guide', default=False, action='store_true', help="whether to use counterfactual guide (default: %(default)s)")
        parser.add_argument(
            '--cf-elbo-type', default=-1, choices=[-1, 0, 1, 2],
            help="-1: randomly select per batch, 0: shuffle thickness, 1: shuffle intensity, 2: shuffle both (default: %(default)s)")
        parser.add_argument('--annealing-epochs', default=50, type=int, help="anneal kl div in z for this # epochs (default: %(default)s)")
        parser.add_argument('--min-annealing-factor', default=0.2, type=float, help="anneal kl div in z starting here (default: %(default)s)")
        parser.add_argument('--max-annealing-factor', default=1.0, type=float, help="anneal kl div in z ending here (default: %(default)s)")
        parser.add_argument('--tracegraph-elbo', default=False, action='store_true', help="use tracegraph elbo (much more computationally expensive) (default: %(default)s)")
        return parser


EXPERIMENT_REGISTRY[SVIExperiment.__name__] = SVIExperiment
