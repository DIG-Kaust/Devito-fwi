__all__ = [
    "gradient_descent",
    "barzilai_borwein_gradient_descent",
]

import time
import numpy as np

from typing import Callable, Optional, Tuple
from pylops.utils.typing import NDArray


def gradient_descent(
    loss_grad: Callable, 
    x0: NDArray, 
    stepsize: float, 
    niter: Optional[int] = 100, 
    args: Optional[Tuple] = None, 
    callback: Optional[Callable] = None, 
    show: Optional[bool] = False,
) -> NDArray:
    """Gradient descent
    
    Optimize function with gradient descent algorithm with fixed step-size

    Parameters
    ----------
    loss_grad : :obj:`func`
        Function taking ``x`` plus any other number of arguments ``args`` as input and 
        returning the loss and gradient
    x0 : :obj:`numpy.ndarray`
        Initial guess (must be a 1d-array)
    stepsize : :obj:`float`
        Step-size
    niter : :obj:`int`, optional
        Number of iterations
    args : :obj:`tuple`, optional
        Additional arguments to pass to the ``loss_grad`` function
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
            Display setup log
        
    Returns
    -------
    x : :obj:`numpy.ndarray`
        Solution

    """

    if show:
        tstart = time.time()
        print('Gradient descent \n'
              '---------------------------------------------------------\n'
              'f: %s\n'
              'stepsize = %s\tniter = %d\n' % (loss_grad, str(stepsize), niter))
        head = '   Itn       x[0]          f'
        print(head)

    x = x0.copy()
    for iiter in range(niter):
        # compute loss and gradient
        f, g = loss_grad(x, *args)
        
        # update model
        x = x - stepsize * g

        # run callback
        if callback is not None:
            callback(x)

        # show iteration logger
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                msg = '%6g  %12.5e  %10.3e' % \
                      (iiter + 1, np.real(x[0]), f)
                print(msg)

    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x


def barzilai_borwein_gradient_descent(
    loss_grad: Callable, 
    x0: NDArray, 
    stepsize: float, 
    steptype: Optional[str] = 'short', 
    epsilon: Optional[float] = 1e-8,
    niter: Optional[int] = 100, 
    args: Optional[Tuple] = None, 
    callback: Optional[Callable] = None, 
    show: Optional[bool] = False,
) -> NDArray:
    """Gradient descent with Barzilai-Borwein step-size
    
    Optimize function with gradient descent algorithm with Barzilai-Borwein step-size
    
    Parameters
    ----------
    loss_grad : :obj:`func`
        Function taking ``x`` plus any other number of arguments ``args`` as input and 
        returning the loss and gradient
    x0 : :obj:`numpy.ndarray`
        Initial guess (must be a 1d-array)
    stepsize : :obj:`float`
        Step-size of first iteration (all subsequent step-sizes will be obtained using the 
        Barzilai-Borwein update rule) 
    steptype : :obj:`str`, optional
        Type of Barzilai-Borwein step-size to compute. It can be either
        'short' for the short-step or 'long' for the long-step. Default is 'short'.
    epsilon : :obj:`float`, optional
        A small value added to the denominator for numerical stability. Default is 1e-8.
    niter : :obj:`int`, optional
        Number of iterations
    args : :obj:`tuple`, optional
        Additional arguments to pass to the ``loss_grad`` function
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
            Display setup log
        
    Returns
    -------
    x : :obj:`numpy.ndarray`
        Solution
        
    """
    def _bb_step(x, xold, g, gold, steptype, epsilon):
        dx = x - xold
        dg = g - gold
        if steptype not in ['short', 'long']:
            raise ValueError("steptype must be 'short' or 'long'")

        if steptype == 'short':
            numerator = np.dot(dx, dg)
            denominator = np.dot(dg, dg)
        elif steptype == 'long':
            numerator = np.dot(dx, dx)
            denominator = np.dot(dx, dg)
        denominator += epsilon
        stepsize = numerator / denominator
        return stepsize

    if show:
        tstart = time.time()
        print('Barzilai-Borwein Gradient descent \n'
              '---------------------------------------------------------\n'
              'f: %s\n'
              'stepsize = %s\tsteptype = %s\n'
              'epsilon = %s\tniter = %d\n' % (loss_grad, str(stepsize), steptype, epsilon, niter))
        head = '   Itn       x[0]          f          step'
        print(head)

    x = x0.copy()
    g = np.zeros_like(x)
    
    for iiter in range(niter):
      
        # compute loss and gradient
        f, g = loss_grad(x, *args)
        
        # compute BB step
        if iiter > 0:
           stepsize = _bb_step(x, xold, g, gold, steptype, epsilon)

        # store previous model and gradient
        xold = x.copy()
        gold = g.copy()

        # update model
        x = x - stepsize * g
        
        # run callback
        if callback is not None:
            callback(x)

        # show iteration logger
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                msg = '%6g  %12.5e  %10.3e  %10.3e' % \
                      (iiter + 1, np.real(x[0]), f, stepsize)
                print(msg)

    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x