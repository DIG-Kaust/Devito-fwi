__all__ = ["NonlinearOperator"]


import numpy as np

from typing import Optional
from pylops.utils.typing import DTypeLike, NDArray
from pylops.linearoperator import _get_dtype


class NonlinearOperator():
    r"""Nonlinear function.

    Template class for a generic nonlinear function :math:`f`, which a user
    must subclass and implement the following methods:

    - ``loss``: a method evaluating the generic function :math:`f`
    - ``grad``: a method evaluating the gradient of the generic function
      :math:`f`
    - ``loss_grad``: a method evaluating both the generic function :math:`f` 
      and its gradient
    
    Parameters
    ----------
    size : :obj:`int`
        Size of the input vector
    dtype : :obj:`str`, optional
        Type of elements in input array.
    
    """
    def __init__(
            self, 
            size: int, 
            dtype: Optional[DTypeLike] = "float32",
    ) -> None:
        self.size = size
        self.dtype = dtype

    def __call__(self, x, *args):
        return self.loss(x, *args)

    def loss(self, x, *args):
        raise NotImplementedError('The method loss has not been implemented.'
                                  'Refer to the documentation for details on '
                                  'how to subclass this operator.')
    def grad(self, x, *args):
        raise NotImplementedError('The method grad has not been implemented.'
                                  'Refer to the documentation for details on '
                                  'how to subclass this operator.')
    def loss_grad(self, x, **kwargs):
        loss, grad = self.loss(x, **kwargs), self.grad(x, **kwargs)
        return loss, grad
    
    def __add__(self, Op):
        Opsum = _SumNonlinearOperator(self, Op)
        return Opsum


class _SumNonlinearOperator(NonlinearOperator):
    def __init__(
        self,
        A: NonlinearOperator,
        B: NonlinearOperator,
    ) -> None:
        if not isinstance(A, NonlinearOperator) or not isinstance(B, NonlinearOperator):
            raise ValueError("both operands have to be a NonlinearOperator")
        if A.size != B.size:
            raise ValueError("cannot add %r and %r: shape mismatch" % (A, B))
        self.args = (A, B)
        super(_SumNonlinearOperator, self).__init__(
            dtype=_get_dtype([A, B]), size=A.size
        )

    def loss(self, x: NDArray, kwargs1={}, kwargs2={}) -> NDArray:
        return self.args[0].loss(x, **kwargs1) + self.args[1].loss(x, **kwargs2)

    def grad(self, x: NDArray, kwargs1={}, kwargs2={}) -> NDArray:
        return self.args[0].grad(x, **kwargs1) + self.args[1].grad(x, **kwargs2)

    def loss_grad(self, x: NDArray, kwargs1={}, kwargs2={}) -> NDArray:
        return self.args[0].loss_grad(x, **kwargs1) + self.args[1].loss_grad(x, **kwargs2)