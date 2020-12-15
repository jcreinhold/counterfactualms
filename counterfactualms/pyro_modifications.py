from torch import nn
from pyro.distributions.transforms import ConditionalSpline
from pyro.nn import DenseNN


def conditional_spline(input_dim, context_dim, hidden_dims=None, count_bins=8, bound=3.0, order='linear'):
    # change from pyro by using leaky relu instead of relu

    if hidden_dims is None:
        hidden_dims = [input_dim * 10, input_dim * 10]

    if order == 'linear':
        net = DenseNN(context_dim,
                      hidden_dims,
                      param_dims=[input_dim * count_bins,
                                  input_dim * count_bins,
                                  input_dim * (count_bins - 1),
                                  input_dim * count_bins],
                      nonlinearity=nn.LeakyReLU(0.1))

    elif order == 'quadratic':
        net = DenseNN(context_dim,
                      hidden_dims,
                      param_dims=[input_dim * count_bins,
                                  input_dim * count_bins,
                                  input_dim * (count_bins - 1)],
                      nonlinearity=nn.LeakyReLU(0.1))

    else:
        raise ValueError("Keyword argument 'order' must be one of ['linear', 'quadratic'], but '{}' was found!".format(order))

    return ConditionalSpline(net, input_dim, count_bins, bound=bound, order=order)


