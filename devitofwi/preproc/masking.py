import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt


def timespace_mask(arrival, nt, dt, toff=0., nsmooth=None):
    """Design time-space mask based on arrival times

    """
    ns, nr = arrival.shape

    # create mask based on arrival times
    mask = np.zeros((ns, nr, nt))
    for isrc in range(ns):
        for irec in range(nr):
            it = np.round((arrival[isrc, irec] + toff) / dt).astype('int')
            mask[isrc, irec, it:] = 1.

    # smooth mask along time axis
    if nsmooth is not None:
        mask = filtfilt(np.ones(nsmooth) / nsmooth, 1, mask, axis=-1)

    return mask