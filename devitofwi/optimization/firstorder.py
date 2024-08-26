import time
import numpy as np


def gradient_descent(loss_grad, x0, stepsize, niter=100, args=None, callback=None, show=False):
    """Gradient descent
    
    Optimize function with gradient descent algorithm
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
        f, g = loss_grad(x, *args)
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
