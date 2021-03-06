# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE_nvidia file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from counterfactualms.arch.thirdparty.swish import Swish as SwishFN
from counterfactualms.arch.thirdparty.batchnormswish import BatchNormSwish
from counterfactualms.arch.layers import Conv2d

from collections import OrderedDict

BN_EPS = 1e-5
SYNC_BN = False

OPS = OrderedDict([
    ('res_elu', lambda Cin, Cout, stride: ELUConv(Cin, Cout, 3, stride, 1)),
    ('res_bnelu', lambda Cin, Cout, stride: BNELUConv(Cin, Cout, 3, stride, 1)),
    ('res_bnswish', lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 1)),
    ('res_bnswish5', lambda Cin, Cout, stride: BNSwishConv(Cin, Cout, 3, stride, 2, 2)),
    ('mconv_e6k5g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=5, g=0)),
    ('mconv_e3k5g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=0)),
    ('mconv_e3k5g8', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=3, dil=1, k=5, g=8)),
    ('mconv_e6k11g0', lambda Cin, Cout, stride: InvertedResidual(Cin, Cout, stride, ex=6, dil=1, k=11, g=0)),
])


def average_tensor(t, is_distributed):
    if is_distributed:
        size = float(dist.get_world_size())
        dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
        t.data /= size


def get_skip_connection(Cin, Cout, stride):
    if stride == 1:
        return Identity()
    elif stride == 2:
        return FactorizedReduce(Cin, Cout)
    elif stride == -1:
        return nn.Sequential(UpSample(), Conv2d(Cin, Cout, kernel_size=1))


def norm(t, dim):
    return torch.sqrt(torch.sum(t * t, dim))


def logit(t):
    return torch.log(t) - torch.log(1 - t)


def act(t):
    # The following implementation has lower memory.
    return SwishFN.apply(t)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return act(x)

@torch.jit.script
def normalize_weight_jit(log_weight_norm, weight):
    n = torch.exp(log_weight_norm)
    wn = torch.sqrt(torch.sum(weight * weight, dim=[1, 2, 3]))   # norm(w)
    weight = n * weight / (wn.view(-1, 1, 1, 1) + 1e-5)
    return weight


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SyncBatchNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SyncBatchNorm, self).__init__()
        self.bn = nn.SyncBatchNorm(*args, **kwargs)

    def forward(self, x):
        # Sync BN only works with distributed data parallel with 1 GPU per process. I don't use DDP, so I need to let
        # Sync BN to know that I have 1 gpu per process.
        self.bn.ddp_gpu_size = 1
        return self.bn(x)


# quick switch between multi-gpu, single-gpu batch norm
def get_batchnorm(*args, **kwargs):
    if SYNC_BN:
        return SyncBatchNorm(*args, **kwargs)
    else:
        return nn.BatchNorm2d(*args, **kwargs)


class ELUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(ELUConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.conv_0 = Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x):
        out = F.elu(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class BNELUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(BNELUConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn = get_batchnorm(C_in, eps=BN_EPS, momentum=0.05)
        self.conv_0 = Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x):
        x = self.bn(x)
        out = F.elu(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out


class BNSwishConv(nn.Module):
    """ReLU + Conv2d + BN."""

    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, dilation=1):
        super(BNSwishConv, self).__init__()
        self.upsample = stride == -1
        stride = abs(stride)
        self.bn_act = BatchNormSwish(C_in, eps=BN_EPS, momentum=0.05)
        self.conv_0 = Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=True, dilation=dilation)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): of size (B, C_in, H, W)
        """
        out = self.bn_act(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv_0(out)
        return out



class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.conv_1 = Conv2d(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_2 = Conv2d(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_3 = Conv2d(C_in, C_out // 4, 1, stride=2, padding=0, bias=True)
        self.conv_4 = Conv2d(C_in, C_out - 3 * (C_out // 4), 1, stride=2, padding=0, bias=True)

    def forward(self, x):
        out = act(x)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:, :, 1:, 1:])
        conv3 = self.conv_3(out[:, :, :, 1:])
        conv4 = self.conv_4(out[:, :, 1:, :])
        out = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        return out


class UpSample(nn.Module):
    def __init__(self):
        super(UpSample, self).__init__()
        pass

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class EncCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(EncCombinerCell, self).__init__()
        self.cell_type = cell_type
        # Cin = Cin1 + Cin2
        self.conv = Conv2d(Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        x2 = self.conv(x2)
        out = x1 + x2
        return out


# original combiner
class DecCombinerCell(nn.Module):
    def __init__(self, Cin1, Cin2, Cout, cell_type):
        super(DecCombinerCell, self).__init__()
        self.cell_type = cell_type
        self.conv = Conv2d(Cin1 + Cin2, Cout, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x1, x2):
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out


class ConvBNSwish(nn.Module):
    def __init__(self, Cin, Cout, k=3, stride=1, groups=1, dilation=1):
        padding = dilation * (k - 1) // 2
        super(ConvBNSwish, self).__init__()

        self.conv = nn.Sequential(
            Conv2d(Cin, Cout, k, stride, padding, groups=groups, bias=False, dilation=dilation, use_weight_norm=False),
            BatchNormSwish(Cout, eps=BN_EPS, momentum=0.05)  # drop in replacement for BN + Swish
        )

    def forward(self, x):
        return self.conv(x)


class SE(nn.Module):
    def __init__(self, Cin, Cout):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se


class InvertedResidual(nn.Module):
    def __init__(self, Cin, Cout, stride, ex, dil, k, g):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2, -1]

        hidden_dim = int(round(Cin * ex))
        self.use_res_connect = self.stride == 1 and Cin == Cout
        self.upsample = self.stride == -1
        self.stride = abs(self.stride)
        groups = hidden_dim if g == 0 else g

        layers0 = [nn.UpsamplingNearest2d(scale_factor=2)] if self.upsample else []
        layers = [get_batchnorm(Cin, eps=BN_EPS, momentum=0.05),
                  ConvBNSwish(Cin, hidden_dim, k=1),
                  ConvBNSwish(hidden_dim, hidden_dim, stride=self.stride, groups=groups, k=k, dilation=dil),
                  Conv2d(hidden_dim, Cout, 1, 1, 0, bias=False, use_weight_norm=False),
                  get_batchnorm(Cout, momentum=0.05)]

        layers0.extend(layers)
        self.conv = nn.Sequential(*layers0)

    def forward(self, x):
        return self.conv(x)
