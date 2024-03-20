import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def timespace_mask(arrival, nt, dt, toff=0., smooth=5):
    """Design time-space mask based on arrival time

    """
    ns, nr = arrival.shape

    mask = np.zeros((ns, nr, nt))
    for isrc in range(ns):
        for irec in range(nr):
            it = np.round((arrival[isrc, irec] + toff) / dt).astype('int')
            mask[isrc, irec, it:] = 1.

    return mask