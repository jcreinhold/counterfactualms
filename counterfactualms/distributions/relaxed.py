import torch
from pyro.distributions import (
    Independent, RelaxedBernoulliStraightThrough
)
from pyro.distributions.torch import RelaxedOneHotCategorical  # noqa: F401
from torch import nn
from torch.distributions.utils import clamp_probs, broadcast_all
from counterfactualms.distributions.deep import DeepConditional


class DeepRelaxedBernoulli(DeepConditional):
    def __init__(self, backbone:nn.Module, temperature:float=2./3.):
        super().__init__()
        self.backbone = backbone
        self.temperature = temperature

    def forward(self, z):
        logits = self.backbone(z)
        return logits

    def predict(self, z) -> Independent:
        logits = self(z)
        temperature = torch.tensor(self.temperature, device=z.device, requires_grad=False)
        event_ndim = len(logits.shape[1:])  # keep only batch dimension
        return RelaxedBernoulliStraightThrough(temperature, logits=logits).to_event(event_ndim)


class DeepRelaxedOneHotCategoricalStraightThrough2D(DeepConditional):
    def __init__(self, backbone: nn.Module, temperature:float=2./3.):
        super().__init__()
        self.backbone = backbone
        self.temperature = temperature

    def forward(self, z):
        logits = self.backbone(z)
        return logits

    def predict(self, z) -> Independent:
        logits = self(z)
        temperature = torch.tensor(self.temperature, device=z.device, requires_grad=False)
        # keep only batch dimension; have to subtract 1 b/c way relaxedonehotcategorical setup
        event_ndim = len(logits.shape[1:]) - 1
        return RelaxedOneHotCategoricalStraightThrough2D(temperature, logits=logits).to_event(event_ndim-1)


class RelaxedOneHotCategorical2D(RelaxedOneHotCategorical):
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniforms = clamp_probs(torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device))
        gumbels = -((-(uniforms.log())).log())
        scores = (self.logits + gumbels) / self.temperature
        return scores - scores.logsumexp(dim=1, keepdim=True)

    def log_prob(self, value):
        K = self._categorical._num_events
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        log_scale = (torch.full_like(self.temperature, float(K)).lgamma() -
                     self.temperature.log().mul(-(K - 1)))
        score = logits - value.mul(self.temperature)
        score = (score - score.logsumexp(dim=1, keepdim=True)).sum((1,2,3))
        return score + log_scale


class RelaxedOneHotCategoricalStraightThrough2D(RelaxedOneHotCategorical2D):
    event_dim = 3
    def rsample(self, sample_shape=torch.Size()):
        soft_sample = super().rsample(sample_shape)
        soft_sample = clamp_probs(soft_sample)
        hard_sample = QuantizeCategorical2D.apply(soft_sample)
        return hard_sample

    def log_prob(self, value):
        value = getattr(value, '_unquantize', value)
        return super().log_prob(value)


class QuantizeCategorical2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, soft_value):
        argmax = soft_value.max(1)[1]
        hard_value = torch.zeros_like(soft_value)
        hard_value._unquantize = soft_value
        if argmax.dim() < hard_value.dim():
            argmax = argmax.unsqueeze(1)
        return hard_value.scatter_(1, argmax, 1)

    @staticmethod
    def backward(ctx, grad):
        return grad


if __name__ == "__main__":
    net = DeepRelaxedBernoulli(nn.Conv2d(2,2,1), 1)
    x = torch.randn(5, 2, 28, 28)
    out = net.predict(x)
    samp = out.rsample()
    print('Bernoulli')
    print(samp.shape)
    print(out.batch_shape, out.event_shape)
    print(out.event_dim)

    net = DeepRelaxedOneHotCategoricalStraightThrough2D(nn.Conv2d(2,2,1), 1)
    out = net.predict(x)
    samp = out.rsample()
    print('OneHot2D')
    print(samp.shape)
    print(out.batch_shape, out.event_shape)
    print(out.event_dim)
