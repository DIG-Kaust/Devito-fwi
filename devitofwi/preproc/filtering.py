import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, correlate, freqs, hilbert


def create_filter(nfilt, fmin, fmax, dt, plotflag=False):
    if fmin is None:
        b, a = butter(nfilt, fmax, 'lowpass', analog=True)
        sos = butter(nfilt, fmax, 'lowpass', fs=1 / dt, output='sos')
    elif fmax is None:
        b, a = butter(nfilt, fmin, 'highpass', analog=True)
        sos = butter(nfilt, fmin, 'highpass', fs=1 / dt, output='sos')
    else:
        b, a = butter(nfilt, [fmin, fmax], 'bandpass', analog=True)
        sos = butter(nfilt, [fmin, fmax], 'bandpass', fs=1 / dt, output='sos')

    if plotflag:
        w, h = freqs(b, a)
        plt.semilogx(w, 20 * np.log10(abs(h)), 'k', lw=2)
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [radians / second]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(fmax, color='green')  # cutoff frequency

    return b, a, sos


def apply_filter(sos, inp):
    filtered = sosfiltfilt(sos, inp, axis=-1)
    return filtered


def filter_data(nfilt, fmin, fmax, dt, inp, plotflag=False):
    """Filter data

    Apply Butterworth  band-pass filter to data

    Parameters
    ----------
    nfilt : :obj:`int`
        Size of filter
    fmin : :obj:`float`
        Minimum frequency in Hz
    fmax : :obj:`float`
        Maximum frequency in Hz
    dt : :obj:`float`
        Time sampling in s
    inp : :obj:`numpy.ndarray`
        Data of size `nx x nt`

    Returns
    -------
    b : :obj:`numpy.ndarray`
        Filter numerator coefficients
    b : :obj:`numpy.ndarray`
        Filter denominator coefficients
    sos : :obj:`numpy.ndarray`
        Filter sos
    filtered : :obj:`numpy.ndarray`
        Filtered data of size `nx x nt`

    """
    b, a, sos = create_filter(nfilt, fmin, fmax, dt, plotflag=plotflag)
    filtered = apply_filter(sos, inp)

    return b, a, sos, filtered


class Filter():
    """Filtering

    Define a sequence of filters to apply to a dataset/wavelet based
    on a list of cut-off frequencies and filter lengths

    Parameters
    ----------
    freqs : :obj:`list`
        Cut-off frequencies in Hz
    nfilt : :obj:`int`
        Size of filters
    dt : :obj:`float`
        Time sampling in s
    plotflag : :obj:`bool`, Optional
        Plot flag (if ``True`` the frequency response of the filters will be visualized

    """
    def __init__(self, freqs, nfilts, dt, plotflag=False):
        self.freqs = freqs
        self.nfilts = nfilts
        self.dt = dt
        self.plotflag = plotflag
        self.filters = self._create_filters()

    def _create_filters(self):
        filters = []

        for freq, nfilt in zip(self.freqs, self.nfilts):
            filters.append(create_filter(nfilt, None, freq, self.dt, plotflag=self.plotflag)[-1])
        return filters

    def apply_filter(self, inp, ifilt=0):
        return apply_filter(self.filters[ifilt], inp)

    def find_optimal_t0(self, inp, pad=400, thresh=1e-2):
        """Find optimal padding

        Identify optimal padding to avoid any filtered signal to become acausal. To be used when designing the filters
        to choose how much the wavelet and observed data must be padded
        """
        inppad = np.pad(inp, (pad, pad))
        itmax = np.argmax(np.abs(inppad))
        it0 = np.where(np.abs(inppad[:itmax]) < thresh * np.abs(inppad[itmax]))[0][-1]
        for ifilt in range(len(self.filters)):
            inpfilt = apply_filter(self.filters[ifilt], inppad)
            inpfiltenv = np.abs(hilbert(inpfilt))
            it0filt = np.where(np.abs(inpfiltenv[:itmax]) < thresh * inpfiltenv[itmax])[0][-1]
            it0 = min(it0, it0filt)
        optimalpad = pad - it0
        return optimalpad