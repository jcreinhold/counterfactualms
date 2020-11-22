import pyro
from pyro.nn import pyro_method, DenseNN
from pyro.distributions import Normal, Bernoulli, Uniform, TransformedDistribution # noqa: F401
from pyro.distributions.conditional import ConditionalTransformedDistribution
import torch

from counterfactualms.distributions.transforms.affine import ConditionalAffineTransform
from counterfactualms.experiments.medical.calabresi.sem_vi.base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalVISEM(BaseVISEM):
    # number of context dimensions for decoder (5 b/c brain vol, ventricle vol, lesion vol, slice number)
    context_dim = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # total_brain_volume flow
        total_brain_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.total_brain_volume_flow_components = ConditionalAffineTransform(context_nn=total_brain_volume_net, event_dim=0)
        self.total_brain_volume_flow_transforms = [
            self.total_brain_volume_flow_components, self.total_brain_volume_flow_constraint_transforms
        ]

        # total_ventricle_volume flow
        total_ventricle_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.total_ventricle_volume_flow_components = ConditionalAffineTransform(context_nn=total_ventricle_volume_net, event_dim=0)
        self.total_ventricle_volume_flow_transforms = [
            self.total_ventricle_volume_flow_components, self.total_ventricle_volume_flow_constraint_transforms
        ]

        # total_lesion_volume flow
        total_lesion_volume_net = DenseNN(3, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.total_lesion_volume_flow_components = ConditionalAffineTransform(context_nn=total_lesion_volume_net, event_dim=0)
        self.total_lesion_volume_flow_transforms = [
            self.total_lesion_volume_flow_components, self.total_lesion_volume_flow_constraint_transforms
        ]

        # brain_volume flow
        brain_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.brain_volume_flow_components = ConditionalAffineTransform(context_nn=brain_volume_net, event_dim=0)
        self.brain_volume_flow_transforms = [
            self.brain_volume_flow_components, self.brain_volume_flow_constraint_transforms
        ]

        # ventricle_volume flow
        ventricle_volume_net = DenseNN(3, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
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
        _ = self.sex_logits
        sex = pyro.sample('sex', sex_dist)

        slice_number_dist = Uniform(low=self.slice_number_low, high=self.slice_number_high).to_event(1)
        _ = self.slice_number_low
        _ = self.slice_number_high
        slice_number = pyro.sample('slice_number', slice_number_dist)

        age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        age_dist = TransformedDistribution(age_base_dist, self.age_flow_transforms)
        age = pyro.sample('age', age_dist)
        age_ = self.age_flow_constraint_transforms.inv(age)
        # pseudo call to register with pyro
        _ = self.age_flow_components

        total_brain_context = torch.cat([sex, age_], 1)
        total_brain_volume_base_dist = Normal(self.total_brain_volume_base_loc, self.total_brain_volume_base_scale).to_event(1)
        total_brain_volume_dist = ConditionalTransformedDistribution(total_brain_volume_base_dist, self.total_brain_volume_flow_transforms).condition(total_brain_context)
        total_brain_volume = pyro.sample('total_brain_volume', total_brain_volume_dist)
        _ = self.total_brain_volume_flow_components
        total_brain_volume_ = self.total_brain_volume_flow_constraint_transforms.inv(total_brain_volume)

        total_ventricle_context = torch.cat([age_, total_brain_volume_], 1)
        total_ventricle_volume_base_dist = Normal(self.total_ventricle_volume_base_loc, self.total_ventricle_volume_base_scale).to_event(1)
        total_ventricle_volume_dist = ConditionalTransformedDistribution(total_ventricle_volume_base_dist, self.total_ventricle_volume_flow_transforms).condition(total_ventricle_context)  # noqa: E501
        total_ventricle_volume = pyro.sample('total_ventricle_volume', total_ventricle_volume_dist)
        _ = self.total_ventricle_volume_flow_components
        total_ventricle_volume_ = self.total_ventricle_volume_flow_constraint_transforms.inv(total_ventricle_volume)

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

        total_lesion_context = torch.cat([total_brain_volume_, duration_, score_], 1)
        total_lesion_volume_base_dist = Normal(self.total_lesion_volume_base_loc, self.total_lesion_volume_base_scale).to_event(1)
        total_lesion_volume_dist = ConditionalTransformedDistribution(total_lesion_volume_base_dist, self.total_lesion_volume_flow_transforms).condition(total_lesion_context)
        total_lesion_volume = pyro.sample('total_lesion_volume', total_lesion_volume_dist)
        _ = self.total_lesion_volume_flow_components
        total_lesion_volume_ = self.total_lesion_volume_flow_constraint_transforms.inv(total_lesion_volume)

        brain_context = torch.cat([total_brain_volume_, slice_number], 1)
        brain_volume_base_dist = Normal(self.brain_volume_base_loc, self.brain_volume_base_scale).to_event(1)
        brain_volume_dist = ConditionalTransformedDistribution(brain_volume_base_dist, self.brain_volume_flow_transforms).condition(brain_context)
        brain_volume = pyro.sample('brain_volume', brain_volume_dist)
        _ = self.brain_volume_flow_components
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)

        ventricle_context = torch.cat([brain_volume_, total_ventricle_volume_, slice_number], 1)
        ventricle_volume_base_dist = Normal(self.ventricle_volume_base_loc, self.ventricle_volume_base_scale).to_event(1)
        ventricle_volume_dist = ConditionalTransformedDistribution(ventricle_volume_base_dist, self.ventricle_volume_flow_transforms).condition(ventricle_context)  # noqa: E501
        ventricle_volume = pyro.sample('ventricle_volume', ventricle_volume_dist)
        _ = self.ventricle_volume_flow_components

        lesion_context = torch.cat([brain_volume_, total_lesion_volume_, slice_number], 1)
        lesion_volume_base_dist = Normal(self.lesion_volume_base_loc, self.lesion_volume_base_scale).to_event(1)
        lesion_volume_dist = ConditionalTransformedDistribution(lesion_volume_base_dist, self.lesion_volume_flow_transforms).condition(lesion_context)
        lesion_volume = pyro.sample('lesion_volume', lesion_volume_dist)
        _ = self.lesion_volume_flow_components

        return dict(age=age, sex=sex, ventricle_volume=ventricle_volume, brain_volume=brain_volume,
                    lesion_volume=lesion_volume, total_ventricle_volume=total_ventricle_volume,
                    total_brain_volume=total_brain_volume, total_lesion_volume=total_lesion_volume,
                    duration=duration, score=score, slice_number=slice_number)

    # no arguments because model is called with condition decorator
    @pyro_method
    def model(self):
        obs = self.pgm_model()

        ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(obs['ventricle_volume'])
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(obs['brain_volume'])
        lesion_volume_ = self.lesion_volume_flow_constraint_transforms.inv(obs['lesion_volume'])

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))
        latent = torch.cat([z, ventricle_volume_, brain_volume_, lesion_volume_, obs['slice_number']], 1)

        x_dist = self._get_transformed_x_dist(latent)
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

            hidden = torch.cat([hidden, ventricle_volume_, brain_volume_, lesion_volume_, obs['slice_number']], 1)
            latent_dist = self.latent_encoder.predict(hidden)
            z = pyro.sample('z', latent_dist)

        return z


MODEL_REGISTRY[ConditionalVISEM.__name__] = ConditionalVISEM
