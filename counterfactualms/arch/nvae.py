import numpy as np
from torch import nn
from torch.nn.utils import spectral_norm

from counterfactualms.arch.thirdparty.utils import get_arch_cells
from counterfactualms.arch.thirdparty.neural_operations import Swish, Conv2D
from counterfactualms.arch.thirdparty.cells import Cell
from counterfactualms.arch.thirdparty.batchnormswish import BatchNormSwish


class Encoder(nn.Module):
    def __init__(self, filters=(16,32,64,128), latent_dim:int=100,
                 input_size=(1,224,224), arch='res_mbconv'):
        super().__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.arch_instance = get_arch_cells(arch)
        self.arch = arch

        layers = []

        n_resolutions = len(filters)
        filters = (1,) + tuple(filters)
        for ci, co in zip(filters, filters[1:]):
            cell_type = 'normal_enc'
            arch = self.arch_instance[cell_type]
            layers += [Cell(ci, ci, cell_type=cell_type, arch=arch, use_se=True)]
            cell_type = 'down_enc'
            arch = self.arch_instance[cell_type]
            layers += [Cell(ci, co, cell_type=cell_type, arch=arch, use_se=True)]
            cur_channels = co

        self.cnn = nn.Sequential(*layers)

        self.intermediate_shape = np.array(input_size) // (2 ** n_resolutions)
        self.intermediate_shape[0] = cur_channels

        self.fc = nn.Sequential(
            nn.Linear(np.prod(self.intermediate_shape), latent_dim),
            BatchNormSwish(latent_dim, momentum=0.05)
        )

    def forward(self, x):
        x = self.cnn(x).view(-1, np.prod(self.intermediate_shape))
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, filters=(128,64,32,16), latent_dim:int=128,
                 output_size=(1,192,192), arch='res_mbconv'):
        super().__init__()

        self.filters = filters
        self.latent_dim = latent_dim
        self.arch_instance = get_arch_cells(arch)
        self.arch = arch

        self.intermediate_shape = np.array(output_size) // (2 ** (len(filters) - 1))
        self.intermediate_shape[0] = filters[0]
        self.fc = nn.Linear(latent_dim, np.prod(self.intermediate_shape), bias=False)

        layers = []

        cur_channels = filters[0]
        for c in filters[1:]:
            cell_type = 'normal_dec'
            arch = self.arch_instance[cell_type]
            layers += [Cell(cur_channels, cur_channels, cell_type=cell_type, arch=arch, use_se=True)]
            cell_type = 'up_dec'
            arch = self.arch_instance[cell_type]
            layers += [Cell(cur_channels, c, cell_type=cell_type, arch=arch, use_se=True)]
            cur_channels = c

        layers += [nn.Sequential(Swish(), spectral_norm(Conv2D(cur_channels, 1, 1, 1)))]

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x).view(-1, *self.intermediate_shape)
        return self.cnn(x)
