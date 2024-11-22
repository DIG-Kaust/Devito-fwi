__all__ = ["XCorrTorch", ]


try:
    import torch
    from pylops import TorchOperator
except:
    print('torch not available, install it '
          'to be able to use L2Torch...')


class XCorrTorch():
    r"""Pearson Correlation coefficient using Torch and AD.

    Computes the Pearson correlation coefficient between traces defined as: :math:`R(\mathbf{x}) =
    \mathbf{Op}\mathbf{x} \cdot \mathbf{b} / (||\mathbf{Op}\mathbf{x}||_2 ||\mathbf{b}||_2)` using Torch and leveraging
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
    nt : :obj:`numpy.ndarray`
        Number of time samples in ``b``
    
    """

    def __init__(self, Op, b, nr, nt):
        self.Op = Op
        self.b = b
        self.nr, self.nt = nr, nt

    def __call__(self, x, i):
        b = torch.from_numpy(self.b[i].reshape(self.nt, self.nr))
        bnorm = b / torch.linalg.norm(b, dim=0)
        if self.Op is not None:
            self.x = torch.from_numpy(x).requires_grad_()
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op
            Opx = TorchOperator(Op).apply(self.x).reshape(self.nt, self.nr)
            xnorm = Opx /torch.linalg.norm(Opx, dim=0)
            f = - torch.sum(torch.mul(bnorm, xnorm))
        else:
            self.x = torch.from_numpy(x).reshape(self.nt, self.nr).requires_grad_()
            xnorm = self.x /torch.linalg.norm(self.x, dim=0)
            f = - torch.sum(torch.mul(bnorm, xnorm))
        self.f = f
        return f.item()

    def grad(self, x, i):
        self.f.backward()
        g = self.x.grad.detach().numpy()
        return g
