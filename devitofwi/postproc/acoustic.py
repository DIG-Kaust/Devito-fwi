import numpy as np
from scipy.ndimage import gaussian_filter


def create_mask(m, value):
    """Mask

    Create mask based on cut-off ``value``. When the mask is applied
    to a vector, this removes all values before such cut-off.

    Parameters
    ----------
    m : :obj:`float`
        Model
    value : :obj:`float`
        Cut-off value

    Returns
    -------
    mask : :obj:`float`
        Mask

    """
    mask = m > value
    mask = mask.astype(int)
    return mask


class PostProcessVP():
    """Postprocess loss and gradient

    This method applies a series of post-processing steps to the loss and gradient computed in
    :func:`devitofwi.waveengine.acoustic.AcousticWave2D` propagator. More specifically:

    - apply scaling to obtain the velocity gradient (note that the acoustic propagator of Devito
      naturally computes the gradient of the acoustic wave equation for the slowness squares.
    - smooth gradient (optional)
    - apply mask to gradient (optional)
    - apply scaling to loss and gradient (optional)

    Parameters
    ----------
    scaling : :obj:`float`, optional
        Scaling to apply to loss and gradient
    sigmas : :obj:`tuple`, optional
        Sigmas of gaussian smoothing to apply to gradient
    mask : :obj:`float`, optional
        Mask of size ``(nx, nz)`` to apply to gradient

    """
    def __init__(self, scaling=1., sigmas=None, mask=None):
        self.scaling = scaling
        self.sigmas = sigmas
        self.mask = mask

    def apply(self, vp, loss, grad):
        """Apply postprocessing

        Parameters
        ----------
        vp : :obj:`numpy.ndarray`
            Velocity model of size ``(nx, nz)``
        loss : :obj:`float`
            Loss function
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, nz)``

        Returns
        -------
        loss : :obj:`float`
            Post-processed loss function
        grad : :obj:`numpy.ndarray`
            Post-processed gradient of size ``(nx, nz)``

        """
        # Scale to obtain velocity gradient
        grad = - grad / (vp ** 3)

        # Mask gradient
        if self.mask is not None:
            grad *= self.mask

        # Smooth gradient
        if self.sigmas is not None:
            grad = gaussian_filter(grad, sigma=self.sigmas)

        # Rescale loss and gradient
        loss /= self.scaling
        grad = grad.astype(np.float64) / self.scaling

        return loss, grad
