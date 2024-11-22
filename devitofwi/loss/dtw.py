__all__ = ["SoftDTW", ]


from pylops import TorchOperator

try:
    import torch
    from tslearn.metrics import SoftDTWLossPyTorch
except:
    print('torch and/or tslearn not available, install them '
          'to be able to use SoftDTW...')


class SoftDTW():
    r"""Soft-DTW.

    Computes the Soft Dynamic Time Warping (Soft-DTW) loss or the
    Soft-DTW divergence loss using ``tslearn`` that leverages Torch and
    Automatic Differentiation for the gradient


    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`
        Data vector (must be passed flattened, it will be reshaped 
        internally using ``nr`` and ``nt``)
    nr : :obj:`int`
        Number of receivers in ``b``
    nt : :obj:`int`
        Number of time samples in ``b``
    gamma : :obj:`float`
        Regularization parameter for the SoftDTW computation.
    normalize : :obj:`bool`, optional
        If True, the Soft-DTW divergence is used, which ensures the loss is positive.
    t_sub : :obj:`int`, optional
        Time subsampling factor to reduce memory requirements when computing the loss. Defaults to 1 (no subsampling).
    device : :obj:`str`, optional
        The device to compute the loss on, typically 'cpu' or 'cuda' for GPU acceleration. Defaults to 'cpu'.

    """

    def __init__(self, Op, b, nr, nt, gamma, normalize=False, t_sub=1, device='cpu'):
        self.Op = Op
        self.b = b
        self.nr, self.nt = nr, nt
        self.t_sub = t_sub
        self.device = device
        self.soft_dtw_loss = SoftDTWLossPyTorch(gamma=gamma, normalize=normalize)
        

    def __call__(self, x, i):
        b = torch.from_numpy(self.b[i].reshape(self.nt, self.nr)).T[:, ::self.t_sub, None].to(self.device)
        if self.Op is not None:
            self.x = torch.from_numpy(x).requires_grad_()
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op
            Opx = TorchOperator(Op).apply(self.x).reshape(self.nt, self.nr).T[:, ::self.t_sub, None].to(self.device)
        else:
            self.x = torch.from_numpy(x).reshape(self.nt, self.nr).requires_grad_().T[:, ::self.t_sub, None].to(self.device)
            Opx = self.x

        f = self.soft_dtw_loss(Opx, b).mean()
        self.f = f
        return f.item()

    def grad(self, x, i):
        self.f.backward()
        g = self.x.grad.detach().numpy().T
        return g
