import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt


class TimeSpaceMasking():
    """Time-space masking

    Create and apply time-space mask to data based on a arrival time curve

    Parameters
    ----------
    arrival : :obj:`numpy.ndarray`
        Arrival time curve of size :math:`ns \times nr`
    nt : :obj:`int`
        Number of time samples
    dt : :obj:`float`
        Time sampling
    toff : :obj:`float`, optional
        Time offset to apply to arrival time curve
    nsmooth : :obj:`int`, optional
        Number of samples for smoothing function to apply to mask
        along the time axis

    """
    def __init__(self, arrival, nt, dt, toff=0., nsmooth=None):
        self.nt = nt
        self.dt = dt
        self.toff = toff
        self.nsmooth = nsmooth
        self.mask = self._create_mask(arrival)

    def _create_mask(self, arrival):
        """Create mask
        """
        ns, nr = arrival.shape

        # create mask based on arrival times
        mask = np.zeros((ns, nr, self.nt))
        for isrc in range(ns):
            for irec in range(nr):
                it = np.round((arrival[isrc, irec] + self.toff) / self.dt).astype('int')
                mask[isrc, irec, it:] = 1.

        # smooth mask along time axis
        if self.nsmooth is not None:
            mask = filtfilt(np.ones(self.nsmooth) / self.nsmooth, 1, mask, axis=-1)

        return 1 - mask

    def apply(self, data):
        return self.mask * data