from torch import nn
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