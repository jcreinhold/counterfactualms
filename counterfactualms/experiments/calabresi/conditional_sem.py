import pyro
from pyro.nn import pyro_method, DenseNN
from pyro.distributions import Normal, Bernoulli, Uniform, TransformedDistribution # noqa: F401
from pyro.distributions.conditional import ConditionalTransformedDistribution
import torch

from counterfactualms.distributions.transforms.affine import ConditionalAffineTransform
from counterfactualms.experiments.calabresi.base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalVISEM(BaseVISEM):
    # number of context dimensions for decoder (3 b/c brain vol, ventricle vol, lesion vol)
    context_dim = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # brain_volume flow
        brain_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.brain_volume_flow_components = ConditionalAffineTransform(context_nn=brain_volume_net, event_dim=0)
        self.brain_volume_flow_transforms = [
            self.brain_volume_flow_components, self.brain_volume_flow_constraint_transforms
        ]

        # ventricle_volume flow
        ventricle_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.ventricle_volume_flow_components = ConditionalAffineTransform(context_nn=ventricle_volume_net, event_dim=0)
        self.ventricle_volume_flow_transforms = [
            self.ventricle_volume_flow_components, self.ventricle_volume_flow_constraint_transforms
        ]

        # lesion_volume flow
        lesion_volume_net = DenseNN(3, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.lesion_volume_flow_components = ConditionalAffineTransform(context_nn=lesion_volume_net, event_dim=0)
        self.lesion_volume_flow_transforms = [
            self.lesion_volume_flow_components, self.lesion_volume_flow_constraint_transforms
        ]
        # slice_brain_volume flow
        slice_brain_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.slice_brain_volume_flow_components = ConditionalAffineTransform(context_nn=slice_brain_volume_net, event_dim=0)
        self.slice_brain_volume_flow_transforms = [
            self.slice_brain_volume_flow_components, self.slice_brain_volume_flow_constraint_transforms
        ]

        # slice_ventricle_volume flow
        slice_ventricle_volume_net = DenseNN(3, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.slice_ventricle_volume_flow_components = ConditionalAffineTransform(context_nn=slice_ventricle_volume_net, event_dim=0)
        self.slice_ventricle_volume_flow_transforms = [
            self.slice_ventricle_volume_flow_components, self.slice_ventricle_volume_flow_constraint_transforms
        ]

        # slice_lesion_volume flow
        slice_lesion_volume_net = DenseNN(3, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.slice_lesion_volume_flow_components = ConditionalAffineTransform(context_nn=slice_lesion_volume_net, event_dim=0)
        self.slice_lesion_volume_flow_transforms = [
            self.slice_lesion_volume_flow_components, self.slice_lesion_volume_flow_constraint_transforms
        ]

        # duration flow
        duration_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.duration_flow_components = ConditionalAffineTransform(context_nn=duration_net, event_dim=0)
        self.duration_flow_transforms = [
            self.duration_flow_components, self.duration_flow_constraint_transforms
        ]

        # score flow
        score_net = DenseNN(1, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.score_flow_components = ConditionalAffineTransform(context_nn=score_net, event_dim=0)
        self.score_flow_transforms = [
            self.score_flow_components, self.score_flow_constraint_transforms
        ]

    @pyro_method
    def pgm_model(self):
        sex_dist = Bernoulli(logits=self.sex_logits).to_event(1)
        # pseudo call to register with pyro
        _ = self.sex_logits
        sex = pyro.sample('sex', sex_dist)

        slice_number_base_dist = Normal(self.slice_number_base_loc, self.slice_number_base_scale).to_event(1)
        slice_number_dist = TransformedDistribution(slice_number_base_dist, self.slice_number_flow_transforms)
        slice_number = pyro.sample('slice_number', slice_number_dist)
        slice_number_ = self.slice_number_flow_constraint_transforms.inv(slice_number)
        _ = self.slice_number_flow_components

        age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        age_dist = TransformedDistribution(age_base_dist, self.age_flow_transforms)
        age = pyro.sample('age', age_dist)
        age_ = self.age_flow_constraint_transforms.inv(age)
        _ = self.age_flow_components

        brain_context = torch.cat([sex, age_], 1)
        brain_volume_base_dist = Normal(self.brain_volume_base_loc, self.brain_volume_base_scale).to_event(1)
        brain_volume_dist = ConditionalTransformedDistribution(brain_volume_base_dist, self.brain_volume_flow_transforms).condition(brain_context)
        brain_volume = pyro.sample('brain_volume', brain_volume_dist)
        _ = self.brain_volume_flow_components
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)

        ventricle_context = torch.cat([age_, brain_volume_], 1)
        ventricle_volume_base_dist = Normal(self.ventricle_volume_base_loc, self.ventricle_volume_base_scale).to_event(1)
        ventricle_volume_dist = ConditionalTransformedDistribution(ventricle_volume_base_dist, self.ventricle_volume_flow_transforms).condition(ventricle_context)  # noqa: E501
        ventricle_volume = pyro.sample('ventricle_volume', ventricle_volume_dist)
        _ = self.ventricle_volume_flow_components
        ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(ventricle_volume)

        duration_context = torch.cat([sex, age_], 1)
        duration_base_dist = Normal(self.duration_base_loc, self.duration_base_scale).to_event(1)
        duration_dist = ConditionalTransformedDistribution(duration_base_dist, self.duration_flow_transforms).condition(duration_context)  # noqa: E501
        duration = pyro.sample('duration', duration_dist)
        _ = self.duration_flow_components
        duration_ = self.duration_flow_constraint_transforms.inv(duration)

        score_context = torch.cat([duration_], 1)
        score_base_dist = Normal(self.score_base_loc, self.score_base_scale).to_event(1)
        score_dist = ConditionalTransformedDistribution(score_base_dist, self.score_flow_transforms).condition(score_context)  # noqa: E501
        score = pyro.sample('score', score_dist)
        _ = self.score_flow_components
        score_ = self.score_flow_constraint_transforms.inv(score)

        lesion_context = torch.cat([brain_volume_, duration_, score_], 1)
        lesion_volume_base_dist = Normal(self.lesion_volume_base_loc, self.lesion_volume_base_scale).to_event(1)
        lesion_volume_dist = ConditionalTransformedDistribution(lesion_volume_base_dist, self.lesion_volume_flow_transforms).condition(lesion_context)
        lesion_volume = pyro.sample('lesion_volume', lesion_volume_dist)
        _ = self.lesion_volume_flow_components
        lesion_volume_ = self.lesion_volume_flow_constraint_transforms.inv(lesion_volume)

        slice_brain_context = torch.cat([brain_volume_, slice_number_], 1)
        slice_brain_volume_base_dist = Normal(self.slice_brain_volume_base_loc, self.slice_brain_volume_base_scale).to_event(1)
        slice_brain_volume_dist = ConditionalTransformedDistribution(slice_brain_volume_base_dist, self.slice_brain_volume_flow_transforms).condition(slice_brain_context)
        slice_brain_volume = pyro.sample('slice_brain_volume', slice_brain_volume_dist)
        _ = self.slice_brain_volume_flow_components
        slice_brain_volume_ = self.slice_brain_volume_flow_constraint_transforms.inv(slice_brain_volume)

        slice_ventricle_context = torch.cat([slice_brain_volume_, ventricle_volume_, slice_number_], 1)
        slice_ventricle_volume_base_dist = Normal(self.slice_ventricle_volume_base_loc, self.slice_ventricle_volume_base_scale).to_event(1)
        slice_ventricle_volume_dist = ConditionalTransformedDistribution(slice_ventricle_volume_base_dist, self.slice_ventricle_volume_flow_transforms).condition(slice_ventricle_context)  # noqa: E501
        slice_ventricle_volume = pyro.sample('slice_ventricle_volume', slice_ventricle_volume_dist)
        _ = self.slice_ventricle_volume_flow_components

        slice_lesion_context = torch.cat([slice_brain_volume_, lesion_volume_, slice_number_], 1)
        slice_lesion_volume_base_dist = Normal(self.slice_lesion_volume_base_loc, self.slice_lesion_volume_base_scale).to_event(1)
        slice_lesion_volume_dist = ConditionalTransformedDistribution(slice_lesion_volume_base_dist, self.slice_lesion_volume_flow_transforms).condition(slice_lesion_context)
        slice_lesion_volume = pyro.sample('slice_lesion_volume', slice_lesion_volume_dist)
        _ = self.slice_lesion_volume_flow_components

        return dict(age=age, sex=sex, ventricle_volume=ventricle_volume, brain_volume=brain_volume,
                    lesion_volume=lesion_volume, slice_ventricle_volume=slice_ventricle_volume,
                    slice_brain_volume=slice_brain_volume, slice_lesion_volume=slice_lesion_volume,
                    duration=duration, score=score, slice_number=slice_number)

    # no arguments because model is called with condition decorator
    @pyro_method
    def model(self):
        obs = self.pgm_model()

        slice_ventricle_volume_ = self.slice_ventricle_volume_flow_constraint_transforms.inv(obs['slice_ventricle_volume'])
        slice_brain_volume_ = self.slice_brain_volume_flow_constraint_transforms.inv(obs['slice_brain_volume'])
        slice_lesion_volume_ = self.slice_lesion_volume_flow_constraint_transforms.inv(obs['slice_lesion_volume'])

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))
        latent = torch.cat([z, slice_ventricle_volume_, slice_brain_volume_, slice_lesion_volume_], 1)

        x_dist = self._get_transformed_x_dist(latent)
        x = pyro.sample('x', x_dist)

        obs.update(dict(x=x, z=z))
        return obs

    @pyro_method
    def guide(self, obs):
        batch_size = obs['x'].shape[0]
        with pyro.plate('observations', batch_size):
            hidden = self.encoder(obs['x'])

            slice_ventricle_volume_ = self.slice_ventricle_volume_flow_constraint_transforms.inv(obs['slice_ventricle_volume'])
            slice_brain_volume_ = self.slice_brain_volume_flow_constraint_transforms.inv(obs['slice_brain_volume'])
            slice_lesion_volume_ = self.slice_lesion_volume_flow_constraint_transforms.inv(obs['slice_lesion_volume'])

            hidden = torch.cat([hidden, slice_ventricle_volume_, slice_brain_volume_, slice_lesion_volume_], 1)
            latent_dist = self.latent_encoder.predict(hidden)
            z = pyro.sample('z', latent_dist)

        return z


MODEL_REGISTRY[ConditionalVISEM.__name__] = ConditionalVISEM
