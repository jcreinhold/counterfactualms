from functools import partial

import numpy as np
from torch import nn

from counterfactualms.arch.layers import Conv2d, ConvTranspose2d


class Encoder(nn.Module):
    def __init__(self, num_convolutions=1, filters=(16,32,64,128,256), latent_dim:int=100,
                 input_size=(1,128,128), use_weight_norm=False, use_spectral_norm=False):
        super().__init__()

        self.num_convolutions = num_convolutions
        self.filters = filters
        self.latent_dim = latent_dim
        if use_weight_norm and use_spectral_norm:
            raise ValueError('Cannot use both weight norm and spectral norm.')
        self.use_weight_norm = use_weight_norm
        self.use_spectral_norm = use_spectral_norm

        layers = []
        cur_channels = input_size[0]
        for c in filters:
            for _ in range(0, num_convolutions - 1):
                layers += self._conv_layer(cur_channels, c)
                cur_channels = c

            layers += self._down_conv_layer(cur_channels, c)

            cur_channels = c

        self.cnn = nn.Sequential(*layers)

        self.intermediate_shape = np.array(input_size) // (2 ** len(filters))
        self.intermediate_shape[0] = cur_channels

        self.fc = nn.Sequential(
            nn.Linear(np.prod(self.intermediate_shape), latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(.1, inplace=True)
        )

    @property
    def _conv(self):
        return partial(Conv2d, use_weight_norm=self.use_weight_norm, use_spectral_norm=self.use_spectral_norm)

    def _conv_layer(self, ci, co):
        return [self._conv(ci, co, 3, 1, 1, bias=False),
                nn.BatchNorm2d(co, momentum=0.05),
                nn.LeakyReLU(.1, inplace=True)]

    def _down_conv_layer(self, ci, co):
        return [self._conv(ci, co, 4, 2, 1, bias=False),
                nn.BatchNorm2d(co, momentum=0.05),
                nn.LeakyReLU(.1, inplace=True)]

    def forward(self, x):
        x = self.cnn(x).view(-1, np.prod(self.intermediate_shape))
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, num_convolutions=1, filters=(256,128,64,32,16), latent_dim=100,
                 output_size=(1,128,128), upconv=False, use_weight_norm=False, use_spectral_norm=False):
        super().__init__()

        self.num_convolutions = num_convolutions
        self.filters = filters
        self.latent_dim = latent_dim
        self.upconv = upconv
        if use_weight_norm and use_spectral_norm:
            raise ValueError('Cannot use both weight norm and spectral norm.')
        self.use_weight_norm = use_weight_norm
        self.use_spectral_norm = use_spectral_norm

        self.intermediate_shape = np.array(output_size) // (2 ** (len(filters)))
        self.intermediate_shape[0] = filters[0]

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, np.prod(self.intermediate_shape), bias=False),
            nn.BatchNorm1d(np.prod(self.intermediate_shape)),
            nn.LeakyReLU(.1, inplace=True)
        )

        layers = []

        cur_channels = filters[0]
        layers += self._upsample_layer(cur_channels, cur_channels)
        for c in filters[1:]:
            for _ in range(0, num_convolutions - 1):
                layers += self._conv_layer(cur_channels)

            layers += self._upsample_layer(cur_channels, c)
            cur_channels = c

        layers += self._conv_layer(cur_channels)
        layers += [self._conv(cur_channels, output_size[0], 1, 1, bias=True)]

        self.cnn = nn.Sequential(*layers)

    @property
    def _conv(self):
        return partial(Conv2d, use_weight_norm=self.use_weight_norm, use_spectral_norm=self.use_spectral_norm)

    @property
    def _conv_transpose(self):
        return partial(ConvTranspose2d, use_weight_norm=self.use_weight_norm, use_spectral_norm=self.use_spectral_norm)

    def _conv_layer(self, c):
        return [self._conv(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c, momentum=0.05),
                nn.LeakyReLU(.1, inplace=True)]

    def _upsample_layer(self, ci, co):
        if self.upconv:
            layer = [nn.Upsample(scale_factor=2, mode='nearest'),
                     self._conv(ci, co, kernel_size=5, stride=1, padding=2, bias=False)]
        else:
            layer = [self._conv_transpose(ci, co, kernel_size=4, stride=2, padding=1, bias=False)]
        layer += [nn.BatchNorm2d(co, momentum=0.05),
                  nn.LeakyReLU(.1, inplace=True)]
        return layer

    def forward(self, x):
        x = self.fc(x).view(-1, *self.intermediate_shape)
        return self.cnn(x)
