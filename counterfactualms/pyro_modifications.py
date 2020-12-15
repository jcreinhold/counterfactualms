import operator
from functools import partial, reduce

from torch import nn
from pyro.distributions.transforms import AffineCoupling, AffineAutoregressive
from pyro.distributions.transforms import SplineAutoregressive, SplineCoupling
from pyro.distributions.transforms import ConditionalSpline
from pyro.nn import AutoRegressiveNN, DenseNN


def conditional_spline(input_dim, context_dim, hidden_dims=None, count_bins=8, bound=3.0, order='linear',
                       nonlinearity=nn.LeakyReLU(0.1)):
    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    if order == 'linear':
        net = DenseNN(context_dim,
                      hidden_dims,
                      param_dims=[input_dim * count_bins,
                                  input_dim * count_bins,
                                  input_dim * (count_bins - 1),
                                  input_dim * count_bins],
                      nonlinearity=nonlinearity)
    elif order == 'quadratic':
        net = DenseNN(context_dim,
                      hidden_dims,
                      param_dims=[input_dim * count_bins,
                                  input_dim * count_bins,
                                  input_dim * (count_bins - 1)],
                      nonlinearity=nonlinearity)
    else:
        raise ValueError("Keyword argument 'order' must be one of ['linear', 'quadratic'], but '{}' was found!".format(order))
    return ConditionalSpline(net, input_dim, count_bins, bound=bound, order=order)


def spline_coupling(input_dim, split_dim=None, hidden_dims=None, count_bins=8, bound=3.0, nonlinearity=nn.LeakyReLU(0.1)):

    if split_dim is None:
        split_dim = input_dim // 2
    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    net = DenseNN(split_dim,
                 hidden_dims,
                 param_dims=[(input_dim - split_dim) * count_bins,
                             (input_dim - split_dim) * count_bins,
                             (input_dim - split_dim) * (count_bins - 1),
                             (input_dim - split_dim) * count_bins],
                 nonlinearity=nonlinearity)
    return SplineCoupling(input_dim, split_dim, net, count_bins, bound)


def spline_autoregressive(input_dim, hidden_dims=None, count_bins=8, bound=3.0, order='linear', nonlinearity=nn.LeakyReLU(0.1)):
    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]
    param_dims = [count_bins, count_bins, count_bins - 1, count_bins]
    arn = AutoRegressiveNN(input_dim, hidden_dims, param_dims=param_dims, nonlinearity=nonlinearity)
    return SplineAutoregressive(input_dim, arn, count_bins=count_bins, bound=bound, order=order)


def affine_autoregressive(input_dim, hidden_dims=None, nonlinearity=nn.LeakyReLU(0.1), **kwargs):
    if hidden_dims is None:
        hidden_dims = [3 * input_dim + 1]
    arn = AutoRegressiveNN(input_dim, hidden_dims, nonlinearity=nonlinearity)
    return AffineAutoregressive(arn, **kwargs)


def affine_coupling(input_dim, hidden_dims=None, split_dim=None, dim=-1, nonlinearity=nn.LeakyReLU(0.1), **kwargs):
    if not isinstance(input_dim, int):
        if len(input_dim) != -dim:
            raise ValueError('event shape {} must have same length as event_dim {}'.format(input_dim, -dim))
        event_shape = input_dim
        extra_dims = reduce(operator.mul, event_shape[(dim + 1):], 1)
    else:
        event_shape = [input_dim]
        extra_dims = 1
    event_shape = list(event_shape)

    if split_dim is None:
        split_dim = event_shape[dim] // 2
    if hidden_dims is None:
        hidden_dims = [10 * event_shape[dim] * extra_dims]

    hypernet = DenseNN(split_dim * extra_dims,
                       hidden_dims,
                       [(event_shape[dim] - split_dim) * extra_dims,
                        (event_shape[dim] - split_dim) * extra_dims],
                       nonlinearity=nonlinearity)
    return AffineCoupling(split_dim, hypernet, dim=dim, **kwargs)