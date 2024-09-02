__all__ = ["L2",
           "L2Torch",
           ]

import numpy as np
import torch
from pylops import TorchOperator

from devitofwi.nonlinear import NonlinearOperator


class L2(NonlinearOperator):
    r"""L2 Norm.

    Computes the :math:`\ell_2` norm defined as: :math:`\ell_2(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    size : :obj:`int`, optional
        Size of the input vector (needed only when both ``Op`` and
        ``b`` are ``None``)
    
    """
    def __init__(self, Op=None, b=None, size=None, dtype="float32"):
        self.Op = Op
        self.b = b
        if size is None:
            if b is None:
                size = Op.shape[1]
            else:
                size = b.size
        super().__init__(size, dtype)

    def loss(self, x, i):
        if self.Op is not None:
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op
        
        if self.Op is not None and self.b is not None:
            f = (1. / 2.) * (np.linalg.norm(Op @ x - self.b[i]) ** 2)
        elif self.b is not None:
            f = (1. / 2.) * (np.linalg.norm(x - self.b[i]) ** 2)
        elif self.Op is not None:
            f = (1. / 2.) * (np.linalg.norm(Op @ x) ** 2)
        else:
            f = (1. / 2.) * (np.linalg.norm(x) ** 2)
        return f
    
    def grad(self, x, i):
        if self.Op is not None:
            Op = self.Op[i] if isinstance(self.Op, list) else self.Op
        
        if self.Op is not None and self.b is not None:
            g = Op.H @ (Op @ x - self.b[i])
        elif self.b is not None:
            g = (x - self.b[i])
        elif self.Op is not None:
            g = Op.H @ Op @ x
        else:
            g = x
        return g


class L2Torch(NonlinearOperator):
    r"""L2 Norm using Torch and AD.

    Computes the :math:`\ell_2` norm defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2` using Torch and leveraging
    Automatic Differentiation for the gradient

    Parameters
    ----------
    b : :obj:`numpy.ndarray`
        Data vector
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    
    """

    def __init__(self, b, Op=None):
        self.Op = Op
        self.b = b
        super().__init__(b.size if Op is None else self.Op.shape[1])

    def loss(self, x, i):
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
