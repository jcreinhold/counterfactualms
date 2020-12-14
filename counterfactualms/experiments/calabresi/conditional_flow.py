from typing import Mapping

import pyro
from pyro.nn import pyro_method, DenseNN
from pyro.distributions import (
    Normal, Bernoulli, Uniform, TransformedDistribution, MixtureOfDiagNormalsSharedCovariance  # noqa: F401
)
from pyro.distributions.conditional import ConditionalTransformedDistribution
from pyro.distributions.transforms import ConditionalSpline
from pyro.distributions.transforms import spline, permute, iterated
from pyro import poutine
import torch
from torch import nn

from counterfactualms.experiments.calabresi.base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalFlowVISEM(BaseVISEM):
    # number of context dimensions for decoder (3 b/c brain vol, ventricle vol, lesion vol)
    context_dim = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.brain_volume_flow_components = conditional_spline(1, 2, [8, 16])
        self.brain_volume_flow_transforms = [
            self.brain_volume_flow_components, self.brain_volume_flow_constraint_transforms
        ]

        self.ventricle_volume_flow_components = conditional_spline(1, 3, [8, 16])
        self.ventricle_volume_flow_transforms = [
            self.ventricle_volume_flow_components, self.ventricle_volume_flow_constraint_transforms
        ]

        self.lesion_volume_flow_components = conditional_spline(1, 4, [16, 32])
        self.lesion_volume_flow_transforms = [
            self.lesion_volume_flow_components, self.lesion_volume_flow_constraint_transforms
        ]

        self.duration_flow_components = conditional_spline(1, 2, [8, 16])
        self.duration_flow_transforms = [
            self.duration_flow_components, self.duration_flow_constraint_transforms
        ]

        self.score_flow_components = conditional_spline(1, 4, [16, 32])
        self.score_flow_transforms = [
            self.score_flow_components, self.score_flow_constraint_transforms
        ]

        prior_permutations = [permute(self.latent_dim) for _ in range(self.n_prior_flows)]
        self.prior_flow_components = iterated(self.n_prior_flows, spline, self.latent_dim)
        self.prior_flow_transforms = [
            x for c in zip(prior_permutations, self.prior_flow_components) for x in c
        ]

        posterior_permutations = [permute(self.latent_dim) for _ in range(self.n_posterior_flows)]
        self.posterior_flow_components = iterated(self.n_posterior_flows, spline, self.latent_dim)
        self.posterior_flow_transforms = [
            x for c in zip(posterior_permutations, self.posterior_flow_components) for x in c
        ]

    @pyro_method
    def pgm_model(self):
        sex_dist = Bernoulli(logits=self.sex_logits).to_event(1)
        # pseudo call to register with pyro
        _ = self.sex_logits
        sex = pyro.sample('sex', sex_dist, infer=dict(baseline={'use_decaying_avg_baseline': True}))

        age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        age_dist = TransformedDistribution(age_base_dist, self.age_flow_transforms)
        _ = self.age_flow_components
        age = pyro.sample('age', age_dist)
        age_ = self.age_flow_constraint_transforms.inv(age)

        brain_context = torch.cat([sex, age_], 1)
        brain_volume_base_dist = Normal(self.brain_volume_base_loc, self.brain_volume_base_scale).to_event(1)
        brain_volume_dist = ConditionalTransformedDistribution(brain_volume_base_dist, self.brain_volume_flow_transforms).condition(brain_context)
        _ = self.brain_volume_flow_components
        brain_volume = pyro.sample('brain_volume', brain_volume_dist)
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)

        duration_context = torch.cat([sex, age_], 1)
        duration_base_dist = Normal(self.duration_base_loc, self.duration_base_scale).to_event(1)
        duration_dist = ConditionalTransformedDistribution(duration_base_dist, self.duration_flow_transforms).condition(duration_context)  # noqa: E501
        duration = pyro.sample('duration', duration_dist)
        _ = self.duration_flow_components
        duration_ = self.duration_flow_constraint_transforms.inv(duration)

        ventricle_context = torch.cat([age_, brain_volume_, duration_], 1)
        ventricle_volume_base_dist = Normal(self.ventricle_volume_base_loc, self.ventricle_volume_base_scale).to_event(1)
        ventricle_volume_dist = ConditionalTransformedDistribution(ventricle_volume_base_dist, self.ventricle_volume_flow_transforms).condition(ventricle_context)  # noqa: E501
        ventricle_volume = pyro.sample('ventricle_volume', ventricle_volume_dist)
        _ = self.ventricle_volume_flow_components
        ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(ventricle_volume)

        score_context = torch.cat([duration_, sex, age_, ventricle_volume_], 1)
        score_base_dist = Normal(self.score_base_loc, self.score_base_scale).to_event(1)
        score_dist = ConditionalTransformedDistribution(score_base_dist, self.score_flow_transforms).condition(score_context)  # noqa: E501
        score = pyro.sample('score', score_dist)
        _ = self.score_flow_components
        score_ = self.score_flow_constraint_transforms.inv(score)

        lesion_context = torch.cat([brain_volume_, ventricle_volume_, duration_, score_], 1)
        lesion_volume_base_dist = Normal(self.lesion_volume_base_loc, self.lesion_volume_base_scale).to_event(1)
        lesion_volume_dist = ConditionalTransformedDistribution(lesion_volume_base_dist, self.lesion_volume_flow_transforms).condition(lesion_context)
        lesion_volume = pyro.sample('lesion_volume', lesion_volume_dist)
        _ = self.lesion_volume_flow_components

        return dict(age=age, sex=sex, ventricle_volume=ventricle_volume, brain_volume=brain_volume,
                    lesion_volume=lesion_volume, duration=duration, score=score)

    # no arguments because model is called with condition decorator
    @pyro_method
    def model(self):
        obs = self.pgm_model()

        ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(obs['ventricle_volume'])
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(obs['brain_volume'])
        lesion_volume_ = self.lesion_volume_flow_constraint_transforms.inv(obs['lesion_volume'])
        ctx = torch.cat([ventricle_volume_, brain_volume_, lesion_volume_], 1)

        z_base_dist = Normal(self.z_loc, self.z_scale).to_event(1)
        z_dist = TransformedDistribution(z_base_dist, self.prior_flow_transforms)
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
            x = self._add_noise(obs['x'])
            hidden = self.encoder(x)

            ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(obs['ventricle_volume'])
            brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(obs['brain_volume'])
            lesion_volume_ = self.lesion_volume_flow_constraint_transforms.inv(obs['lesion_volume'])
            ctx = torch.cat([ventricle_volume_, brain_volume_, lesion_volume_], 1)
            hidden = torch.cat([hidden, ctx], 1)

            latent_base_dist = self.latent_encoder.predict(hidden)
            latent_dist = TransformedDistribution(latent_base_dist, self.posterior_flow_transforms)
            _ = self.posterior_flow_components
            with poutine.scale(scale=self.annealing_factor):
                z = pyro.sample('z', latent_dist)

        return z


def conditional_spline(input_dim, context_dim, hidden_dims=None, count_bins=8, bound=3.0, order='linear'):

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    if order == 'linear':
        net = DenseNN(context_dim,
                     hidden_dims,
                     param_dims=[input_dim * count_bins,
                                 input_dim * count_bins,
                                 input_dim * (count_bins - 1),
                                 input_dim * count_bins],
                     nonlinearity=nn.LeakyReLU(0.1))

    elif order == 'quadratic':
        net = DenseNN(context_dim,
                     hidden_dims,
                     param_dims=[input_dim * count_bins,
                                 input_dim * count_bins,
                                 input_dim * (count_bins - 1)],
                     nonlinearity=nn.LeakyReLU(0.1))

    else:
        raise ValueError("Keyword argument 'order' must be one of ['linear', 'quadratic'], but '{}' was found!".format(
            order))

    return ConditionalSpline(net, input_dim, count_bins, bound=bound, order=order)


MODEL_REGISTRY[ConditionalFlowVISEM.__name__] = ConditionalFlowVISEM
