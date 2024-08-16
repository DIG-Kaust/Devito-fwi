__all__ = ["Empty",
           ]

import numpy as np


class Empty():
    r"""Empty Norm.
    
    This class implements an empty norm, where the gradient 
    is simply represented by the data vector (and the norm
    itself simply returns 0). To be used only to perform RTM
    (i.e., to have an adjoint source equal to the data)

    Parameters
    ----------
    b : :obj:`numpy.ndarray`
        Data vector
    
    """
    def __init__(self, b):
        self.b = b
        
    def __call__(self, x, i):
        return 0.
    
    def grad(self, x, i):
        return self.b[i]