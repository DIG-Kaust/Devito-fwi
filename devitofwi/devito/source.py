import numpy as np
import matplotlib.pyplot as plt

from examples.seismic.utils import sources, PointSource


class CustomSource(PointSource):
    """Custom source

    This class creates a Devito symbolic object that encapsulates a set of
    sources with a user defined source signal wavelet ``wav``

    Parameters
    ----------
    name : :obj:`str`
        Name for the resulting symbol.
    grid : :obj:`devito.types.grid.Grid`
        The computational domain.
    time_range : :obj:`examples.seismic.source.TimeAxis`
        TimeAxis(start, step, num) object.
    wav : :obj:`numpy.ndarray`
        Wavelet of size
    
    """

    __rkwargs__ = PointSource.__rkwargs__ + ['wav']

    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        kwargs.setdefault('npoint', 1)

        return super().__args_setup__(*args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        super().__init_finalize__(*args, **kwargs)

        self.wav = kwargs.get('wav')

        if not self.alias:
            for p in range(kwargs['npoint']):
                self.data[:, p] = self.wavelet

    @property
    def wavelet(self):
        """
        Return user-provided wavelet
        """
        return self.wav

    def show(self, idx=0, wavelet=None):
        """
        Plot the wavelet of the specified source.

        Parameters
        ----------
        idx : int
            Index of the source point for which to plot wavelet.
        wavelet : ndarray or callable
            Prescribed wavelet instead of one from this symbol.
        """
        wavelet = wavelet or self.data[:, idx]
        plt.figure()
        plt.plot(self.time_values, wavelet)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.tick_params()
        plt.show()
        
        
sources['CustomSource'] = CustomSource

