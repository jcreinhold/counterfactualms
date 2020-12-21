import numpy as np
import pyro
from pyro.nn import pyro_method
from pyro.distributions import (
    Normal, Bernoulli, Uniform, TransformedDistribution  # noqa: F401
)
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro import poutine
import torch
from torch import nn

from counterfactualms.arch.hierarchical import HierarchicalDecoder, HierarchicalEncoder
from counterfactualms.arch.thirdparty.neural_operations import Swish
from counterfactualms.arch.layers import Conv2d
from counterfactualms.distributions.deep import Conv2dIndepNormal
from counterfactualms.experiments.calabresi.base_experiment import MODEL_REGISTRY
from counterfactualms.experiments.calabresi.base_sem_experiment import BaseVISEM
from counterfactualms.utils.pyro_modifications import conditional_spline


class BaseHierarchicalVISEM(BaseVISEM):
    def __init__(self, *args, **kwargs):
        kwargs['n_prior_flows'] = 0
        kwargs['n_posterior_flows'] = 0
        super().__init__(*args, **kwargs)
        self.encoder = HierarchicalEncoder(num_convolutions=self.num_convolutions, filters=self.enc_filters,
                                           input_size=self.img_shape, use_weight_norm=self.use_weight_norm,
                                           use_spectral_norm=self.use_spectral_norm)
        decoder = HierarchicalDecoder(num_convolutions=self.num_convolutions, filters=self.dec_filters,
                                      output_size=self.img_shape, use_weight_norm=self.use_weight_norm,
                                      use_spectral_norm=self.use_spectral_norm, context_dim=self.context_dim)
        self._create_decoder(decoder)
        self.context_attn = nn.ModuleList([])
        self.latent_encoder = nn.ModuleList([])
        del self.z_loc, self.z_scale
        self.intermediate_shapes = self.encoder.intermediate_shapes
        assert all([np.all(eis == dis) for eis, dis in zip(self.intermediate_shapes, decoder.intermediate_shapes[::-1])])
        self.n_levels = len(self.intermediate_shapes)
        for i, z_size in enumerate(self.intermediate_shapes):
            n_latent_channels = z_size[0]
            hidden_dim = max(n_latent_channels // 4, self.context_dim)
            nonlinearity = Swish() if self.use_swish else torch.nn.LeakyReLU(0.1, inplace=True)
            self.context_attn.append(nn.Sequential(
                nn.Linear(self.context_dim, hidden_dim),
                nonlinearity,
                nn.Linear(hidden_dim, n_latent_channels),
                nn.Sigmoid()
            ))
            backbone = nn.Sequential(Conv2d(n_latent_channels, n_latent_channels, 3, padding=1,
                                            use_weight_norm=self.use_weight_norm,
                                            use_spectral_norm=self.use_spectral_norm),
                                     nonlinearity)
            self.latent_encoder.append(Conv2dIndepNormal(backbone, n_latent_channels, n_latent_channels))
            self.register_buffer(f'z_loc_{i}', torch.zeros(z_size.tolist(), requires_grad=False))
            self.register_buffer(f'z_scale_{i}', torch.ones(z_size.tolist(), requires_grad=False))

    @pyro_method
    def infer(self, obs):
        self._check_observation(obs)
        obs_ = obs.copy()
        z = self.infer_z(obs_)
        for i in range(self.n_levels):
            obs_.update({f'z{i}': z[i]})
        exogenous = self.infer_exogenous(obs_)
        for i in range(self.n_levels):
            exogenous[f'z{i}'] = z[i]
        return exogenous

    @pyro_method
    def reconstruct(self, obs, num_particles:int=1):
        self._check_observation(obs)
        z_dists = []
        guide_trace = pyro.poutine.trace(self.guide).get_trace(obs)
        for i in range(self.n_levels):
            z_dists.append(guide_trace.nodes[f'z{i}']['fn'])
        batch_size = obs['x'].shape[0]
        obs_ = {k: v for k, v in obs.items() if k != 'x'}
        recons = []
        for _ in range(num_particles):
            z = []
            for i in range(self.n_levels):
                z.append(pyro.sample(f'z{i}', z_dists[i]))
                obs_.update({f'z{i}': z[-1]})
            recon = pyro.poutine.condition(
                self.sample, data=obs_)(batch_size)
            recons += [recon['x']]
        return torch.stack(recons).mean(0)

    @pyro_method
    def counterfactual(self, obs, condition=None, num_particles:int=1):
        self._check_observation(obs)
        obs_ = obs.copy()
        z_dists = []
        guide_trace = pyro.poutine.trace(self.guide).get_trace(obs_)
        for i in range(self.n_levels):
            z_dists.append(guide_trace.nodes[f'z{i}']['fn'])  # variational posterior
        n = obs_['x'].shape[0]

        counterfactuals = []
        for _ in range(num_particles):
            z = []
            for i in range(self.n_levels):
                z.append(pyro.sample(f'z{i}', z_dists[i]))
                obs_.update({f'z{i}': z[-1]})
            exogenous = self.infer_exogenous(obs_)
            for i in range(self.n_levels):
                exogenous[f'z{i}'] = z[i]
            # condition on these vars if they aren't included in 'do' as they are root nodes
            # and we don't have the exogenous noise for them yet
            if 'sex' not in condition.keys():
                exogenous['sex'] = obs_['sex']
            if 'slice_number' not in condition.keys():
                exogenous['slice_number'] = obs_['slice_number']

            cf = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogenous), data=condition)(n)
            counterfactuals.append(cf)

        return self._cf_dict(counterfactuals)


class ConditionalHierarchicalFlowVISEM(BaseHierarchicalVISEM):
    # number of context dimensions for decoder (4 b/c brain vol, ventricle vol, lesion vol, slice num)
    context_dim = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        nonlinearity = Swish() if self.use_swish else nn.LeakyReLU(0.1)

        self.brain_volume_flow_components = conditional_spline(1, 2, [8, 16], nonlinearity=nonlinearity)
        self.brain_volume_flow_transforms = [
            self.brain_volume_flow_components, self.brain_volume_flow_constraint_transforms
        ]

        self.ventricle_volume_flow_components = conditional_spline(1, 3, [12, 20], nonlinearity=nonlinearity)
        self.ventricle_volume_flow_transforms = [
            self.ventricle_volume_flow_components, self.ventricle_volume_flow_constraint_transforms
        ]

        self.lesion_volume_flow_components = conditional_spline(1, 4, [16, 24], nonlinearity=nonlinearity)
        self.lesion_volume_flow_transforms = [
            self.lesion_volume_flow_components, self.lesion_volume_flow_constraint_transforms
        ]

        self.duration_flow_components = conditional_spline(1, 2, [8, 16], nonlinearity=nonlinearity)
        self.duration_flow_transforms = [
            self.duration_flow_components, self.duration_flow_constraint_transforms
        ]

        self.edss_flow_components = conditional_spline(1, 2, [8, 16], nonlinearity=nonlinearity)
        self.edss_flow_transforms = [
            self.edss_flow_components, self.edss_flow_constraint_transforms
        ]

    @pyro_method
    def pgm_model(self):
        sex_dist = Bernoulli(logits=self.sex_logits).to_event(1)
        # pseudo call to register with pyro
        _ = self.sex_logits
        sex = pyro.sample('sex', sex_dist, infer=dict(baseline={'use_decaying_avg_baseline': True}))

        slice_number_dist = Uniform(self.slice_number_min, self.slice_number_max).to_event(1)
        slice_number = pyro.sample('slice_number', slice_number_dist)

        age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        age_dist = TransformedDistribution(age_base_dist, self.age_flow_transforms)
        _ = self.age_flow_components
        age = pyro.sample('age', age_dist)
        age_ = self.age_flow_constraint_transforms.inv(age)

        duration_context = torch.cat([sex, age_], 1)
        duration_base_dist = Normal(self.duration_base_loc, self.duration_base_scale).to_event(1)
        duration_dist = ConditionalTransformedDistribution(duration_base_dist, self.duration_flow_transforms).condition(duration_context)  # noqa: E501
        duration = pyro.sample('duration', duration_dist)
        _ = self.duration_flow_components
        duration_ = self.duration_flow_constraint_transforms.inv(duration)

        edss_context = torch.cat([sex, duration_], 1)
        edss_base_dist = Normal(self.edss_base_loc, self.edss_base_scale).to_event(1)
        edss_dist = ConditionalTransformedDistribution(edss_base_dist, self.edss_flow_transforms).condition(edss_context)  # noqa: E501
        edss = pyro.sample('edss', edss_dist)
        _ = self.edss_flow_components
        edss_ = self.edss_flow_constraint_transforms.inv(edss)

        brain_context = torch.cat([sex, age_], 1)
        brain_volume_base_dist = Normal(self.brain_volume_base_loc, self.brain_volume_base_scale).to_event(1)
        brain_volume_dist = ConditionalTransformedDistribution(brain_volume_base_dist, self.brain_volume_flow_transforms).condition(brain_context)
        _ = self.brain_volume_flow_components
        brain_volume = pyro.sample('brain_volume', brain_volume_dist)
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)

        ventricle_context = torch.cat([age_, brain_volume_, duration_], 1)
        ventricle_volume_base_dist = Normal(self.ventricle_volume_base_loc, self.ventricle_volume_base_scale).to_event(1)
        ventricle_volume_dist = ConditionalTransformedDistribution(ventricle_volume_base_dist, self.ventricle_volume_flow_transforms).condition(ventricle_context)  # noqa: E501
        ventricle_volume = pyro.sample('ventricle_volume', ventricle_volume_dist)
        _ = self.ventricle_volume_flow_components
        ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(ventricle_volume)

        lesion_context = torch.cat([brain_volume_, ventricle_volume_, duration_, edss_], 1)
        lesion_volume_base_dist = Normal(self.lesion_volume_base_loc, self.lesion_volume_base_scale).to_event(1)
        lesion_volume_dist = ConditionalTransformedDistribution(lesion_volume_base_dist, self.lesion_volume_flow_transforms).condition(lesion_context)
        lesion_volume = pyro.sample('lesion_volume', lesion_volume_dist)
        _ = self.lesion_volume_flow_components

        return dict(age=age, sex=sex, ventricle_volume=ventricle_volume, brain_volume=brain_volume,
                    lesion_volume=lesion_volume, duration=duration, edss=edss, slice_number=slice_number)

    # no arguments because model is called with condition decorator
    @pyro_method
    def model(self):
        obs = self.pgm_model()

        ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(obs['ventricle_volume'])
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(obs['brain_volume'])
        lesion_volume_ = self.lesion_volume_flow_constraint_transforms.inv(obs['lesion_volume'])
        slice_number = obs['slice_number']
        ctx = torch.cat([ventricle_volume_, brain_volume_, lesion_volume_, slice_number], 1)

        z = []
        for i in range(self.n_levels):
            z_loc = getattr(self, f'z_loc_{i}')
            z_scale = getattr(self, f'z_scale_{i}')
            z_dist = Normal(z_loc, z_scale).to_event(3)
            with poutine.scale(scale=self.annealing_factor):
                z.append(pyro.sample(f'z{i}', z_dist))

        x_dist = self._get_transformed_x_dist(z, ctx)  # run decoder
        x = pyro.sample('x', x_dist)

        obs.update(dict(x=x, z=z))
        return obs

    @pyro_method
    def guide(self, obs):
        batch_size = obs['x'].shape[0]
        with pyro.plate('observations', batch_size):
            hidden = self.encoder(obs['x'])

            ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(obs['ventricle_volume'])
            brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(obs['brain_volume'])
            lesion_volume_ = self.lesion_volume_flow_constraint_transforms.inv(obs['lesion_volume'])
            slice_number = obs['slice_number']
            ctx = torch.cat([ventricle_volume_, brain_volume_, lesion_volume_, slice_number], 1)

            z = []
            for i, (latent_enc, ctx_attn) in enumerate(zip(self.latent_encoder,self.context_attn)):
                ctx_ = ctx_attn(ctx).view(batch_size, -1, 1, 1)
                hidden_i = hidden[i] * ctx_
                z_dist = latent_enc.predict(hidden_i)
                with poutine.scale(scale=self.annealing_factor):
                    z.append(pyro.sample(f'z{i}', z_dist))

        return z


MODEL_REGISTRY[ConditionalHierarchicalFlowVISEM.__name__] = ConditionalHierarchicalFlowVISEM
