__all__ = [
    "TorchOperator",
]

import numpy as np
import torch

from functools import partial


class _TorchOperator(torch.autograd.Function):
    """Wrapper class for Devito operators into Torch functions"""

    @staticmethod
    def forward(ctx, x, propagator, devicetorch):
        ctx.propagator = propagator
        ctx.devicetorch = devicetorch

        # bring x to cpu and numpy
        x = x.cpu().detach().numpy()

        # apply forward operator
        loss, grad = ctx.propagator(x)

        # prepare output
        ctx.grad = torch.from_numpy(grad.reshape(x.shape))
        y = torch.from_numpy(np.array(loss)).to(ctx.devicetorch)

        return y

    @staticmethod
    def backward(ctx, y):
        # get the pre-computed gradient
        x = ctx.grad.to(ctx.devicetorch)

        return x, None, None, None


class TorchOperator:
    """Wrap a Devito operator into a Torch function.

    This class can be used to wrap a devitofwi waveengine operator into a
    torch function. Doing so, users can mix native torch functions (e.g.
    basic linear algebra operations, neural networks, etc.) and waveengine
    operators.

    Since the computation of the loss function and gradient share some components
    (e.g., source wavefield), our implementation computes both the loss and gradient in forward and
    stores the gradient to be later outputed when the backward method is called.

    Parameters
    ----------
    prop : :obj:`funct`
        Devito waveengine operator `loss_grad` method
    devicetorch : :obj:`str`, optional
        Device to be assigned the output of the operator to (any Torch-compatible device)
    kwargs_prop : :obj:`dict`, optional
        Keywords arguments to be passed to the `loss_grad` method

    """

    def __init__(self, prop, devicetorch="cpu", kwargs_prop=None):
        self.prop = prop
        self.devicetorch = devicetorch
        self.kwargs_prop = kwargs_prop
        self.Top = _TorchOperator.apply

    def __call__(self, x):
        return self.apply(x)

    def apply(self, x):
        """Apply forward pass to input vector

        Parameters
        ----------
        x : :obj:`torch.Tensor`
            Input array

        Returns
        -------
        y : :obj:`torch.Tensor`
            Output array resulting from the application of the operator to ``x``.

        """
        return self.Top(x, partial(self.prop, **self.kwargs_prop), self.devicetorch)
