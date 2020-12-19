import pyro
from pyro.nn import pyro_method
from pyro.distributions import (
    Normal, Bernoulli, Uniform, TransformedDistribution  # noqa: F401
)
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro import poutine
import torch
from torch import nn

from counterfactualms.arch.thirdparty.neural_operations import Swish
from counterfactualms.experiments.calabresi.base_experiment import MODEL_REGISTRY
from counterfactualms.experiments.calabresi.base_sem_experiment import BaseVISEM
from counterfactualms.utils.pyro_modifications import conditional_spline


class ConditionalFlowVISEM(BaseVISEM):
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

        z_base_dist = Normal(self.z_loc, self.z_scale).to_event(1)
        z_dist = TransformedDistribution(z_base_dist, self.prior_flow_transforms) if self.use_prior_flow else z_base_dist
        _ = self.prior_affine
        _ = self.prior_flow_components
        with poutine.scale(scale=self.annealing_factor):
            z = pyro.sample('z', z_dist)
        latent = torch.cat([z, ctx], 1)

        x_dist = self._get_transformed_x_dist(latent)  # run decoder
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
            hidden = torch.cat([hidden, ctx], 1)

            z_base_dist = self.latent_encoder.predict(hidden)
            z_dist = TransformedDistribution(z_base_dist, self.posterior_flow_transforms) if self.use_posterior_flow else z_base_dist
            _ = self.posterior_affine
            _ = self.posterior_flow_components
            with poutine.scale(scale=self.annealing_factor):
                z = pyro.sample('z', z_dist)

        return z


MODEL_REGISTRY[ConditionalFlowVISEM.__name__] = ConditionalFlowVISEM
