from functools import partial

import numpy as np
import torch
from torch import nn

from counterfactualms.arch.layers import Conv2d, ConvTranspose2d


class HierarchicalEncoder(nn.Module):
    def __init__(self, num_convolutions=3, filters=(16,32,64,128,256), latent_dim=100,
                 input_size=(1,128,128), use_weight_norm=False, use_spectral_norm=False,
                 hierarchical_layers=(1,3,5), div_factor=8):
        super().__init__()

        self.num_convolutions = num_convolutions
        self.filters = filters
        if use_weight_norm and use_spectral_norm:
            raise ValueError('Cannot use both weight norm and spectral norm.')
        self.use_weight_norm = use_weight_norm
        self.use_spectral_norm = use_spectral_norm
        self.hierarchical_layers = hierarchical_layers
        self.div_factor = div_factor

        self.down_layers = nn.ModuleList([])
        self.resolution_layers = nn.ModuleList([])
        self.intermediate_shapes = []
        self.out_layers = nn.ModuleList([])
        cur_channels = input_size[0]
        for i, c in enumerate(filters):
            resolution_layer = []
            for _ in range(0, num_convolutions - 1):
                resolution_layer += self._conv_layer(cur_channels, c)
                cur_channels = c
            self.resolution_layers.append(nn.Sequential(*resolution_layer))
            if i in self.hierarchical_layers:
                out_channels = max(cur_channels // div_factor, 1)
                self.out_layers.append(self._conv(cur_channels, out_channels, 1, bias=True))
                self.intermediate_shapes.append(np.array(input_size) // (2 ** i))
                self.intermediate_shapes[-1][0] = out_channels
            self.down_layers.append(nn.Sequential(*self._down_conv_layer(cur_channels, c)))
            cur_channels = c
        if len(filters) in self.hierarchical_layers:
            self.intermediate_shapes.append(np.array(input_size) // (2 ** len(filters)))
            self.intermediate_shapes[-1][0] = cur_channels

        self.fc = nn.Sequential(
            nn.Linear(np.prod(self.intermediate_shapes[-1]), latent_dim, bias=False),
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
        out = []
        c = 0
        for i, (conv, down) in enumerate(zip(self.resolution_layers, self.down_layers)):
            x = conv(x)
            if i in self.hierarchical_layers:
                out.append(self.out_layers[c](x))
                c += 1
            x = down(x)
        if len(self.filters) in self.hierarchical_layers:
            x = x.view(-1, np.prod(self.intermediate_shapes[-1]))
            out.append(self.fc(x))
        return out


class HierarchicalDecoder(nn.Module):
    def __init__(self, num_convolutions=3, filters=(256,128,64,32,16), latent_dim=100, output_size=(1,128,128),
                 upconv=False, use_weight_norm=False, use_spectral_norm=False, hierarchical_layers=(1,3,5),
                 context_dim=4, div_factor=8):
        super().__init__()

        self.num_convolutions = num_convolutions
        self.filters = filters
        self.upconv = upconv
        if use_weight_norm and use_spectral_norm:
            raise ValueError('Cannot use both weight norm and spectral norm.')
        self.use_weight_norm = use_weight_norm
        self.use_spectral_norm = use_spectral_norm
        self.hierarchical_layers = hierarchical_layers
        hierarchical_layers_ = [h for h in hierarchical_layers if h != len(filters)]
        self.context_dim = context_dim
        self.div_factor = div_factor

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
        for i, c in enumerate(filters[1:], 1):
            resolution_layer = []
            i = (len(filters) - i)
            input_layer = i in hierarchical_layers_
            in_channels = max(cur_channels // div_factor, 1)
            for j in range(0, num_convolutions - 1):
                ci = (in_channels+cur_channels) if j == 0 and input_layer else cur_channels
                resolution_layer += self._conv_layer(ci, cur_channels)
            self.resolution_layers.append(nn.Sequential(*resolution_layer))
            self.context_attention.append(self._attn(cur_channels))
            self.up_layers.append(nn.Sequential(*self._upsample_layer(cur_channels, c)))
            if input_layer:
                self.intermediate_shapes.append(np.array(output_size) // (2 ** i))
                self.intermediate_shapes[-1][0] = in_channels
            cur_channels = c

        final_layer = self._conv_layer(cur_channels, cur_channels)
        final_layer.append(self._conv(cur_channels, output_size[0], 1, 1, bias=True))
        self.final_layer = nn.Sequential(*final_layer)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, np.prod(self.intermediate_shapes[0]), bias=False),
            nn.BatchNorm1d(np.prod(self.intermediate_shapes[0])),
            nn.LeakyReLU(.1, inplace=True)
        )

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
        y = self.fc(x.pop()).view(-1, *self.intermediate_shapes[0])
        y = self.start_up_layer(y) * ctx_attn
        for i, (conv, up, attn) in enumerate(layers, 1):
            i = len(self.filters) - i
            output_layer = i in self.hierarchical_layers
            ctx_attn = attn(ctx).view(batch_size, -1, 1, 1)
            if output_layer:
                y = torch.cat([y, x.pop()], 1)
            y = conv(y) * ctx_attn
            y = up(y)
        y = self.final_layer(y)
        return y


if __name__ == "__main__":
    hl = (1, 2, 3, 4, 5)
    filters = [20, 40, 80, 160, 320]
    div_factor = 80
    img_shape = (3,128,128)
    enc = HierarchicalEncoder(
        hierarchical_layers=hl, filters=filters,
        div_factor=div_factor, input_size=img_shape
    )
    dec = HierarchicalDecoder(
        hierarchical_layers=hl, filters=filters[::-1],
        div_factor=div_factor, output_size=img_shape
    )
    print(enc.intermediate_shapes)
    print(dec.intermediate_shapes)
    ctx = torch.randn(2, 4)
    x = torch.randn(2, *img_shape)
    y = enc(x)
    z = dec(y, ctx)
    assert z.shape == x.shape
    print(enc)
    print(dec)
