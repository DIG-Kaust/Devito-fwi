__all__ = ["L2"]

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsqr as sp_lsqr
from pylops import MatrixMult, Identity, TorchOperator
from pylops.optimization.basic import lsqr
from pylops.utils.backend import get_array_module, get_module_name


class L2():
    r"""L2 Norm.

    Computes the :math:`\ell_2` norm defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    
    """
    def __init__(self, Op=None, b=None):
        self.Op = Op
        self.b = b
        
    def __call__(self, x, i):
        if self.Op is not None and self.b is not None:
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op
            f = (1. / 2.) * (np.linalg.norm(Op @ x - self.b[i]) ** 2)
        elif self.b is not None:
            f = (1. / 2.) * (np.linalg.norm(x - self.b[i]) ** 2)
        else:
            f = (1. / 2.) * (np.linalg.norm(x) ** 2)
        return f
    
    def grad(self, x, i):
        if self.Op is not None and self.b is not None:
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op
            g = Op.H @ (Op @ x - self.b[i])
        elif self.b is not None:
            g = (x - self.b[i])
        else:
            g = x
        return g


class L2Torch():
    r"""L2 Norm using Torch and AD.

    Computes the :math:`\ell_2` norm defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2` using Torch and leveraging
    Automatic Differentiation for the gradient

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector

    """

    def __init__(self, Op=None, b=None):
        self.Op = Op
        self.b = b

    def __call__(self, x, i):
        self.x = torch.from_numpy(x).requires_grad_()
        if self.Op is not None:
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op
            f = (1. / 2.) * (torch.linalg.vector_norm(TorchOperator(Op).apply(self.x) -
                                                      torch.from_numpy(self.b[i])) ** 2)
        else:
            f = (1. / 2.) * (torch.linalg.vector_norm(self.x - torch.from_numpy(self.b[i])) ** 2)
        self.f = f
        return f.item()

    def grad(self, x, i):
        self.f.backward()
        g = self.x.grad.detach().numpy()
        return g
