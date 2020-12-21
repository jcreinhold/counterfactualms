from functools import partial

import numpy as np
import torch
from torch import nn

from counterfactualms.arch.layers import Conv2d, ConvTranspose2d


class HierarchicalEncoder(nn.Module):
    def __init__(self, num_convolutions=3, filters=(16,32,64,128,256), input_size=(1,128,128),
                 use_weight_norm=False, use_spectral_norm=False, hierarchical_layers=(1,3,5)):
        super().__init__()

        self.num_convolutions = num_convolutions
        self.filters = filters
        if use_weight_norm and use_spectral_norm:
            raise ValueError('Cannot use both weight norm and spectral norm.')
        self.use_weight_norm = use_weight_norm
        self.use_spectral_norm = use_spectral_norm
        self.hierarchical_layers = hierarchical_layers

        self.down_layers = nn.ModuleList([])
        self.resolution_layers = nn.ModuleList([])
        self.intermediate_shapes = []
        self.out_layers = nn.ModuleList([])
        cur_channels = 1
        for i, c in enumerate(filters):
            resolution_layer = []
            for _ in range(0, num_convolutions - 1):
                resolution_layer += self._conv_layer(cur_channels, c)
                cur_channels = c
            self.resolution_layers.append(nn.Sequential(*resolution_layer))
            if i in self.hierarchical_layers:
                self.out_layers.append(self._conv(cur_channels, cur_channels, 1, bias=True))
            self.down_layers.append(nn.Sequential(*self._down_conv_layer(cur_channels, c)))
            cur_channels = c
            if i in self.hierarchical_layers:
                self.intermediate_shapes.append(np.array(input_size) // (2 ** i))
                self.intermediate_shapes[-1][0] = cur_channels
        if len(filters) in self.hierarchical_layers:
            self.out_layers.append(self._conv(cur_channels, cur_channels, 1, bias=True))
            self.intermediate_shapes.append(np.array(input_size) // (2 ** len(filters)))
            self.intermediate_shapes[-1][0] = cur_channels

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
        out = []
        c = 0
        for i, (conv, down) in enumerate(zip(self.resolution_layers, self.down_layers)):
            x = conv(x)
            if i in self.hierarchical_layers:
                out.append(self.out_layers[c](x))
                c += 1
            x = down(x)
        if len(self.filters) in self.hierarchical_layers:
            out.append(self.out_layers[-1](x))
        return out


class HierarchicalDecoder(nn.Module):
    def __init__(self, num_convolutions=3, filters=(256,128,64,32,16), output_size=(1,128,128),
                 upconv=False, use_weight_norm=False, use_spectral_norm=False, hierarchical_layers=(1,3,5),
                 context_dim=4):
        super().__init__()

        self.num_convolutions = num_convolutions
        self.filters = filters
        self.upconv = upconv
        if use_weight_norm and use_spectral_norm:
            raise ValueError('Cannot use both weight norm and spectral norm.')
        self.use_weight_norm = use_weight_norm
        self.use_spectral_norm = use_spectral_norm
        self.hierarchical_layers = [h for h in hierarchical_layers if h != len(filters)]
        self.context_dim = context_dim

        self.resolution_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])
        self.intermediate_shapes = []
        self.context_attention = nn.ModuleList([])

        cur_channels = filters[0]
        self.start_context_attention = self._attn(cur_channels)
        self.start_up_layer = nn.Sequential(*self._upsample_layer(cur_channels, cur_channels))
        if len(filters) in hierarchical_layers:
            self.intermediate_shapes.append(np.array(output_size) // (2 ** (len(filters))))
            self.intermediate_shapes[-1][0] = cur_channels
        for i, c in enumerate(filters[1:]):
            resolution_layer = []
            for j in range(0, num_convolutions - 1):
                ci = (2*cur_channels) if j == 0 and i in self.hierarchical_layers else cur_channels
                resolution_layer += self._conv_layer(ci, cur_channels)
            self.resolution_layers.append(nn.Sequential(*resolution_layer))
            self.context_attention.append(self._attn(cur_channels))
            self.up_layers.append(nn.Sequential(*self._upsample_layer(cur_channels, c)))
            if i in self.hierarchical_layers:
                self.intermediate_shapes.append(np.array(output_size) // (2 ** (len(filters)-i-1)))
                self.intermediate_shapes[-1][0] = cur_channels
            cur_channels = c

        final_layer = self._conv_layer(cur_channels, cur_channels)
        final_layer.append(self._conv(cur_channels, 1, 1, 1, bias=True))
        self.final_layer = nn.Sequential(*final_layer)

    @property
    def _conv(self):
        return partial(Conv2d, use_weight_norm=self.use_weight_norm, use_spectral_norm=self.use_spectral_norm)

    @property
    def _conv_transpose(self):
        return partial(ConvTranspose2d, use_weight_norm=self.use_weight_norm, use_spectral_norm=self.use_spectral_norm)

    def _conv_layer(self, ci, co):
        return [self._conv(ci, co, 3, 1, 1, bias=False),
                nn.BatchNorm2d(co, momentum=0.05),
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

    def _attn(self, co):
        hidden_dim = max(co // 4, self.context_dim)
        return nn.Sequential(nn.Linear(self.context_dim, hidden_dim),
                             nn.LeakyReLU(0.1, inplace=True),
                             nn.Linear(hidden_dim, co),
                             nn.Sigmoid())

    def forward(self, x, ctx):
        assert x[0].size(0) == ctx.size(0)
        batch_size = ctx.size(0)
        layers = zip(self.resolution_layers, self.up_layers, self.context_attention)
        ctx_attn = self.start_context_attention(ctx).view(batch_size, -1, 1, 1)
        y = self.start_up_layer(x.pop()) * ctx_attn
        for i, (conv, up, attn) in enumerate(layers):
            ctx_attn = attn(ctx).view(batch_size, -1, 1, 1)
            if i in self.hierarchical_layers:
                y = torch.cat([y, x.pop()], 1)
            y = conv(y) * ctx_attn
            y = up(y)
        y = self.final_layer(y)
        return y


if __name__ == "__main__":
    enc = HierarchicalEncoder()
    dec = HierarchicalDecoder()
    ctx = torch.randn(1, 4)
    x = torch.randn(1, 1, 128, 128)
    y = enc(x)
    z = dec(y, ctx)
    assert z.shape == x.shape
