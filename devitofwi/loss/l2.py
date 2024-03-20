__all__ = ["L2"]

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsqr as sp_lsqr
from pylops import MatrixMult, Identity
from pylops.optimization.basic import lsqr
from pylops.utils.backend import get_array_module, get_module_name


class L2():
    r"""L2 Norm.

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
            #import matplotlib.pyplot as plt
            #plt.imshow((Op @ x).reshape(4107, 300), vmin=-.01, vmax=.01, cmap='gray')
            #plt.axis('tight')
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
