# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from counterfactualms.arch.thirdparty.neural_operations import OPS, get_skip_connection, SE
from counterfactualms.arch.thirdparty.neural_ar_operations import  ARInvertedResidual, MixLogCDFParam, mix_log_cdf_flow
from counterfactualms.arch.thirdparty.neural_ar_operations import ELUConv as ARELUConv
from counterfactualms.arch.thirdparty.utils import get_stride_for_cell_type


class Cell(nn.Module):
    def __init__(self, Cin, Cout, cell_type, arch, use_se=True):
        super(Cell, self).__init__()
        self.cell_type = cell_type

        stride = get_stride_for_cell_type(self.cell_type)
        self.skip = get_skip_connection(Cin, Cout, stride)
        self.use_se = use_se
        self._num_nodes = len(arch)
        self._ops = nn.ModuleList()
        for i in range(self._num_nodes):
            stride = get_stride_for_cell_type(self.cell_type) if i == 0 else 1
            C = Cin if i == 0 else Cout
            primitive = arch[i]
            op = OPS[primitive](C, Cout, stride)
            self._ops.append(op)

        if self.use_se:
            self.se = SE(Cout, Cout)

    def forward(self, s):
        # skip branch
        skip = self.skip(s)
        for i in range(self._num_nodes):
            s = self._ops[i](s)
        s = self.se(s) if self.use_se else s
        return skip + 0.1 * s


class CellAR(nn.Module):
    def __init__(self, num_z, num_ftr, num_c, arch, mirror):
        super(CellAR, self).__init__()
        assert num_c % num_z == 0

        self.cell_type = 'ar_nn'

        # s0 will the random samples
        ex = 6
        self.conv = ARInvertedResidual(num_z, num_ftr, ex=ex, mirror=mirror)

        self.use_mix_log_cdf = False
        if self.use_mix_log_cdf:
            self.param = MixLogCDFParam(num_z, num_mix=3, num_ftr=self.conv.hidden_dim, mirror=mirror)
        else:
            # 0.1 helps bring mu closer to 0 initially
            self.mu = ARELUConv(self.conv.hidden_dim, num_z, kernel_size=1, padding=0, masked=True, zero_diag=False,
                                weight_init_coeff=0.1, mirror=mirror)

    def forward(self, z, ftr):
        s = self.conv(z, ftr)

        if self.use_mix_log_cdf:
            logit_pi, mu, log_s, log_a, b = self.param(s)
            new_z, log_det = mix_log_cdf_flow(z, logit_pi, mu, log_s, log_a, b)
        else:
            mu = self.mu(s)
            new_z = (z - mu)
            log_det = torch.zeros_like(new_z)

        return new_z, log_det


class PairedCellAR(nn.Module):
    def __init__(self, num_z, num_ftr, num_c, arch=None):
        super(PairedCellAR, self).__init__()
        self.cell1 = CellAR(num_z, num_ftr, num_c, arch, mirror=False)
        self.cell2 = CellAR(num_z, num_ftr, num_c, arch, mirror=True)

    def forward(self, z, ftr):
        new_z, log_det1 = self.cell1(z, ftr)
        new_z, log_det2 = self.cell2(new_z, ftr)

        log_det1 += log_det2
        return new_z, log_det1
