from typing import List
import torch
from torch.nn import functional as F
from pyro.distributions import (
    Bernoulli, LowRankMultivariateNormal, Beta, Gamma,  # noqa: F401
    Independent, MultivariateNormal, Normal, TorchDistribution,
    MixtureOfDiagNormalsSharedCovariance
)
from torch import nn

from counterfactualms.arch.layers import Conv2d
from counterfactualms.distributions.params import MixtureParams


class DeepConditional(nn.Module):
    def predict(self, x: torch.Tensor) -> TorchDistribution:
        raise NotImplementedError


class _DeepIndepNormal(DeepConditional):
    def __init__(self, backbone:nn.Module, mean_head:nn.Module, logstd_head:nn.Module, logstd_ref:float=0.):
        super().__init__()
        self.backbone = backbone
        self.mean_head = mean_head
        self.logstd_head = logstd_head
        self.logstd_ref = logstd_ref

    def forward(self, x, ctx=None):
        h = self.backbone(x) if ctx is None else self.backbone(x, ctx)
        mean = self.mean_head(h)
        logstd = self.logstd_head(h)
        return mean, logstd

    def predict(self, x, ctx=None) -> Independent:
        mean, logstd = self(x, ctx)
        # use softplus b/c behaves like e^x for (large) neg numbers
        # but doesn't blow up for large positive numbers (leading to nans)
        std = F.softplus(logstd + self.logstd_ref) + 1e-5
        event_ndim = len(mean.shape[1:])  # keep only batch dimension
        return Normal(mean, std).to_event(event_ndim)


class DeepIndepNormal(_DeepIndepNormal):
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int):
        super().__init__(
            backbone=backbone,
            mean_head=nn.Linear(hidden_dim, out_dim),
            logstd_head=nn.Linear(hidden_dim, out_dim)
        )


def _conv_layer(ci, co, use_weight_norm=False, use_spectral_norm=False):
    return [Conv2d(ci, co, 3, 1, 1, bias=False,
                   use_weight_norm=use_weight_norm,
                   use_spectral_norm=use_spectral_norm),
            nn.BatchNorm2d(co, momentum=0.05),
            nn.LeakyReLU(.1, inplace=True)]


def _create_head(head_filters, out_channels:int=1, **kwargs):
    layers = [_conv_layer(hi, ho, **kwargs) for hi, ho in zip(head_filters, head_filters[1:])]
    layers = [h for l in layers for h in l]
    layers.append(nn.Conv2d(head_filters[-1], out_channels, 1))
    return nn.Sequential(*layers)


class Conv2dIndepNormal(_DeepIndepNormal):
    def __init__(self, backbone:nn.Module, hidden_channels:List[int],
                 out_channels:int=1, logstd_ref:float=-5., **kwargs):
        super().__init__(
            backbone=backbone,
            mean_head=_create_head(hidden_channels, out_channels, **kwargs),
            logstd_head=_create_head(hidden_channels, out_channels, **kwargs),
            logstd_ref=logstd_ref
        )


class Conv3dIndepNormal(_DeepIndepNormal):
    def __init__(self, backbone: nn.Module, hidden_channels: int, out_channels: int = 1):
        super().__init__(
            backbone=backbone,
            mean_head=nn.Conv3d(hidden_channels, out_channels=out_channels, kernel_size=1),
            logstd_head=nn.Conv3d(hidden_channels, out_channels=out_channels, kernel_size=1)
        )

class _DeepIndepMixtureNormal(DeepConditional):
    def __init__(self, backbone:nn.Module, mean_head:nn.ModuleList, logstd_head:nn.Module, component_head:nn.Module):
        super().__init__()
        self.backbone = backbone
        self.mean_head = mean_head
        self.logstd_head = logstd_head
        self.component_head = component_head
        def _init_normal(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, 0., 0.02)
                nn.init.constant_(m.bias, 0.)
        self.apply(_init_normal)

    def forward(self, x):
        h = self.backbone(x)
        mean = torch.stack([mh(h) for mh in self.mean_head],1)
        logstd = self.logstd_head(h)
        component = self.component_head(h)
        return mean, logstd, component

    def predict(self, x) -> Independent:
        mean, logstd, component = self(x)
        std = F.softplus(logstd) + 1e-5
        component = torch.log_softmax(component, dim=-1)
        event_ndim = 0
        return MixtureOfDiagNormalsSharedCovariance(mean, std, component).to_event(event_ndim)


class DeepIndepMixtureNormal(_DeepIndepMixtureNormal):
    def __init__(self, backbone:nn.Module, hidden_dim:int, out_dim:int, n_comp:int=2):
        super().__init__(
            backbone=backbone,
            mean_head=nn.ModuleList([nn.Linear(hidden_dim, out_dim) for _ in range(n_comp)]),
            logstd_head=nn.Linear(hidden_dim, out_dim),
            component_head=nn.Linear(hidden_dim, n_comp)
        )


def _assemble_tril(diag: torch.Tensor, lower_vec: torch.Tensor) -> torch.Tensor:
    dim = diag.shape[-1]
    L = torch.diag_embed(diag)  # L is lower-triangular
    i, j = torch.tril_indices(dim, dim, offset=-1)
    L[..., i, j] = lower_vec
    return L


class DeepMultivariateNormal(DeepConditional):
    def __init__(self, backbone: nn.Module, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.backbone = backbone
        cov_lower_dim = (latent_dim * (latent_dim - 1)) // 2
        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.lower_head = nn.Linear(hidden_dim, cov_lower_dim)
        self.logdiag_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h)
        diag = F.softplus(self.logdiag_head(h))
        lower = self.lower_head(h)
        scale_tril = _assemble_tril(diag, lower)
        return mean, scale_tril

    def predict(self, x) -> MultivariateNormal:
        mean, scale_tril = self(x)
        return MultivariateNormal(mean, scale_tril=scale_tril)


class DeepLowRankMultivariateNormal(DeepConditional):
    def __init__(self, backbone: nn.Module, hidden_dim: int, latent_dim: int, rank: int):
        super().__init__()
        self.backbone = backbone

        self.latent_dim = latent_dim
        self.rank = rank

        self.mean_head = nn.Linear(hidden_dim, latent_dim)
        self.factor_head = nn.Linear(hidden_dim, latent_dim * rank)
        self.logdiag_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.backbone(x)
        mean = self.mean_head(h)
        diag = F.softplus(self.logdiag_head(h))
        factors = self.factor_head(h).view(x.shape[0], self.latent_dim, self.rank)

        return mean, diag, factors

    def predict(self, x) -> LowRankMultivariateNormal:
        mean, diag, factors = self(x)
        return LowRankMultivariateNormal(mean, factors, diag)


class MixtureSIN(DeepConditional):
    def __init__(self, encoder: DeepConditional, mixture_params: MixtureParams):
        super().__init__()
        self.encoder = encoder
        self.mixture_params = mixture_params

    def predict(self, data) -> TorchDistribution:
        potentials = self.encoder.predict(data)
        mixture = self.mixture_params.get_distribution()
        posteriors = mixture.posterior(potentials)  # q(latents | data)
        return posteriors


class _DeepIndepGamma(DeepConditional):
    def __init__(self, backbone: nn.Module, rate_head: nn.Module, conc_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.rate_head = nn.Sequential(rate_head, nn.Softplus())
        self.conc_head = nn.Sequential(conc_head, nn.Softplus())

    def forward(self, x):
        h = self.backbone(x)
        rate = self.rate_head(h)
        conc = self.conc_head(h)
        return rate, conc

    def predict(self, x) -> Independent:
        rate, conc = self(x)
        event_ndim = len(rate.shape[1:])  # keep only batch dimension
        return Gamma(rate, conc).to_event(event_ndim)


class DeepIndepGamma(_DeepIndepGamma):
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int):
        super().__init__(
            backbone=backbone,
            rate_head=nn.Linear(hidden_dim, out_dim),
            conc_head=nn.Linear(hidden_dim, out_dim)
        )


class _DeepIndepBeta(DeepConditional):
    def __init__(self, backbone: nn.Module, alpha_head: nn.Module, beta_head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.alpha_head = nn.Sequential(alpha_head, nn.Softplus())
        self.beta_head = nn.Sequential(beta_head, nn.Softplus())

    def forward(self, x):
        h = self.backbone(x)
        alpha = self.alpha_head(h)
        beta = self.beta_head(h)
        return alpha, beta

    def predict(self, x) -> Independent:
        alpha, beta = self(x)
        event_ndim = len(alpha.shape[1:])  # keep only batch dimension
        return Beta(alpha, beta).to_event(event_ndim)


class DeepIndepBeta(_DeepIndepBeta):
    def __init__(self, backbone: nn.Module, hidden_dim: int, out_dim: int):
        super().__init__(
            backbone=backbone,
            alpha_head=nn.Linear(hidden_dim, out_dim),
            beta_head=nn.Linear(hidden_dim, out_dim)
        )


class Conv2dIndepBeta(_DeepIndepBeta):
    def __init__(self, backbone: nn.Module, hidden_channels: int = 1, out_channels: int = 1):
        super().__init__(
            backbone=backbone,
            alpha_head=nn.Conv2d(hidden_channels, out_channels=out_channels, kernel_size=1),
            beta_head=nn.Conv2d(hidden_channels, out_channels=out_channels, kernel_size=1)
        )


class Conv3dIndepBeta(_DeepIndepBeta):
    def __init__(self, backbone: nn.Module, hidden_channels: int = 1, out_channels: int = 1):
        super().__init__(
            backbone=backbone,
            alpha_head=nn.Conv3d(hidden_channels, out_channels=out_channels, kernel_size=1),
            beta_head=nn.Conv3d(hidden_channels, out_channels=out_channels, kernel_size=1)
        )


class DeepBernoulli(DeepConditional):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, z):
        logits = self.backbone(z)
        return logits

    def predict(self, z) -> Independent:
        logits = self(z)
        event_ndim = len(logits.shape[1:])  # keep only batch dimension
        return Bernoulli(logits=logits).to_event(event_ndim)


if __name__ == '__main__':
    import torch
    from counterfactualms.arch import medical

    hidden_dim = 10
    latent_dim = 10
    encoder = DeepIndepNormal(medical.Encoder(hidden_dim), hidden_dim, latent_dim)
    x = torch.randn(5, 1, 28, 28)
    post = encoder.predict(x)
    print(post.batch_shape, post.event_shape)

    decoder = DeepBernoulli(medical.Decoder(latent_dim))
    latents = post.rsample()
    print(latents.shape)
    recon = decoder.predict(latents)
    print(recon.batch_shape, recon.event_shape)
