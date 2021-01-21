import torch.nn as nn
from torch.nn.utils import spectral_norm, weight_norm


class Conv2d(nn.Module):
    def __init__(self, *args, use_weight_norm=True, use_spectral_norm=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.use_weight_norm = use_weight_norm
        self.use_spectral_norm = use_spectral_norm
        if use_weight_norm:
            self.conv = weight_norm(self.conv)
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)

    def _norm_str(self):
        norm = ''
        if self.use_weight_norm:
            norm +=  f', weight_norm={self.use_weight_norm}'
        if self. use_spectral_norm:
            norm += f', spectral_norm={self.use_spectral_norm}'
        norm += ')'
        return norm

    def __repr__(self):
        return self.conv.__repr__()[:-1] + self._norm_str()

    def __str__(self):
        return self.conv.__str__()[:-1] + self._norm_str()


class ConvTranspose2d(nn.Module):
    def __init__(self, *args, use_weight_norm=True, use_spectral_norm=False, **kwargs):
        super().__init__()
        self.conv = nn.ConvTranspose2d(*args, **kwargs)
        self.use_weight_norm = use_weight_norm
        self.use_spectral_norm = use_spectral_norm
        if use_weight_norm:
            self.conv = weight_norm(self.conv)
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)

    def _norm_str(self):
        norm = ''
        if self.use_weight_norm:
            norm +=  f', weight_norm={self.use_weight_norm}'
        if self. use_spectral_norm:
            norm += f', spectral_norm={self.use_spectral_norm}'
        norm += ')'
        return norm

    def __repr__(self):
        return self.conv.__repr__()[:-1] + self._norm_str()

    def __str__(self):
        return self.conv.__str__()[:-1] + self._norm_str()