__all__ = ["AcousticFWI2D"]

from typing import Optional, Type, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from tqdm.notebook import tqdm
from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike

from examples.seismic import AcquisitionGeometry, Model, Receiver
from examples.seismic.acoustic import AcousticWaveSolver

from devitofwi.devito.source import CustomSource
from devitofwi.preproc.filtering import filter_data, Filter
from devitofwi.waveengine.acoustic import AcousticWave2D


class AcousticFWI2D():
    """Devito Acoustic FWI.

    This class provides a wrapper to run a multi-frequency acoustic FWI job based on user defined parameters.
    Note that if ``frequencies=None``, this reduces to a single-frequency FWI job.

    Parameters
    ----------
    par : :obj:`dict`
        Parameters of acquisition geometry
    vp_init : :obj:`numpy.ndarray`
        Initial velocity model in m/s as starting guess for inversion
    vp_range : :obj:`tuple`, optional
        Velocity range (min, max) to be used in the definition of the propagators (to ensure stable and
        non-dispersive modelling)
    wav : :obj:`numpy.ndarray`, optional
        Wavelet (if provided ``src_type`` and ``f0`` will be ignored
    loss : :obj:`devitofwi.loss.`, optional
        Object implementing a loss function and its gradient
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    firstscaling : :obj:`bool`, optional
        Compute first gradient and scale all gradients by its maximum or not
    lossop : :obj:`pylops.LinearOperator` or :obj:`list`, optional
        Linear operator(s) to be applied to the modelled data within the loss function (whne multiple operators
        are provided, they are applied in order to the different stages of multi-frequency FWI - note that
        ``len(lossop)==len(frequencies)``.
    postprocess : :obj:`funct`, optional
        Function handle applying postprocessing to gradient and loss
    convertvp : :obj:`func`, optional
        Function handle that converts the model obtained by the solver in velocity to be used by the propagator
        (if ``None``, it is assumed that the solver itself is working with a velocity model)
    frequencies : :obj:`tuple`, optional
        List of frequencies to be used as maximum frequencies to sequentially filter the observed data
        and wavelet ``wav`` to perform a multi-frequency FWI job
    nfilts : :obj:`tuple`, optional
        Size of filters to sequentially filter the observed data
        and wavelet ``wav`` to perform a multi-frequency FWI job
    nfft : :obj:`int`, optional
        Size of fft used to display the frequency response of the filters alongside with that of the data
    wavpad : :obj:`int`, optional
        Initial padding used to estimate the ideal padding to apply to the wavelet based on the first low-pass filter
        (i.e., first frequency in ``frequencies``).
    kwargs_solver : :obj:`int` or :obj:`tuple`, optional
        Additional keyword arguments to be passed to the solver
    """
    def __init__(self, par,
                 vp_init, vp_range,
                 wav, loss,
                 space_order=4, nbl=20,
                 firstscaling=True, lossop=None, postprocess=None, convertvp=None,
                 frequencies=None, nfilts=None, nfft=2**10, wavpad=700,
                 solver='L-BFGS-B', kwargs_solver=None, callback=None):

        # Save parameters for FWI
        self.par = par
        self.vp_init = vp_init
        self.vp_range = vp_range
        self.wav = wav
        self.loss = loss
        self.space_order = space_order
        self.nbl = nbl
        self.firstscaling = firstscaling
        self.lossop = _value_or_sized_to_tuple(lossop)
        self.postprocess = postprocess
        self.convertvp = convertvp
        self.callback = callback

        # Save parameters for FWI stages
        self.frequencies = _value_or_sized_to_tuple(frequencies)
        self.nfilts = _value_or_sized_to_tuple(nfilts)
        self.nfft = nfft
        self.wavpad = wavpad

        # Save parameters for solver
        self.solver = solver
        self.kwargs_solver = kwargs_solver

        # Model size
        self.shape = (par['nx'], par['nz'])
        self.spacing = (par['dx'], par['dz'])
        self.origin = (par['ox'], par['oz'])

        # Sampling frequency
        self.fs = 1 / par['dt']

        # Axes
        self.x = np.arange(par['nx']) * par['dx'] + par['ox']
        self.z = np.arange(par['nz']) * par['dz'] + par['oz']
        self.t = np.arange(par['nt']) * par['dt'] + par['ot']
        self.tmax = self.t[-1] * 1e3  # in ms

        # Sources
        self.x_s = np.zeros((par['ns'], 2))
        self.x_s[:, 0] = np.arange(par['ns']) * par['ds'] + par['os']
        self.x_s[:, 1] = par['sz']

        # Receivers
        self.x_r = np.zeros((par['nr'], 2))
        self.x_r[:, 0] = np.arange(par['nr']) * par['dr'] + par['or']
        self.x_r[:, 1] = par['rz']


    def run(self, dobs, plotflag=False, vlims=None):
        """FWI Runner

        Run an entire FWI job

        Parameters
        ----------
        dobs : :obj:`numpy.ndarray`
            Observed data of size :math:`n_s \times n_r \times n_t`
        plotflag : :obj:`bool`, optional
            Plot flag. If ``plotflag==True`` various intermediate plots will
            be generated. This is mostly useful for debugging.
        vlims : :obj:`tuple`, optional
            Limits used to plot the velocity models

        """

        # Create reference modelling engine
        amod = AcousticWave2D(self.shape, self.origin, self.spacing,
                              self.x_s[:, 0], self.x_s[:, 1],
                              self.x_r[:, 0], self.x_r[:, 1],
                              0., self.tmax,
                              vprange=self.vp_range,
                              wav=self.wav, f0=self.par['freq'],
                              space_order=self.space_order, nbl=self.nbl)

        # Prepare data and wavelet to allow filtering
        if self.frequencies[0] is not None:
            # Define filter
            if plotflag:
                plt.figure(figsize=(15, 6))
            Filt = Filter(self.frequencies, self.nfilts, amod.geometry.dt * 1e-3, plotflag=plotflag)
            wav = amod.geometry.src.wavelet
            if plotflag:
                f = np.fft.rfftfreq(self.nfft, amod.geometry.dt * 1e-3)
                WAV = np.fft.rfft(wav, self.nfft)
                plt.semilogx(f, 20 * np.log10(np.abs(WAV)) - 28, 'r')

            # Find optimal padding to apply such that lowest freq wavelet does not become acausal
            wavpad = Filt.find_optimal_t0(wav, pad=self.wavpad, thresh=1e-3)

            # Pad wavelet with optimal wavpad
            wav = np.pad(wav, (wavpad, 0))[:-wavpad]

            # Pad observed data accordingly
            dobspad = np.pad(dobs, ((0, 0), (wavpad, 0), (0, 0)))[:, :-wavpad]

            if plotflag:
                d_vmin, d_vmax = np.percentile(np.hstack(dobs).ravel(), [2, 98])

                fig, axs = plt.subplots(1, 3, figsize=(14, 9))
                for ax, ishot in zip(axs, [0, self.par['ns'] // 2, self.par['ns'] - 1]):
                    ax.imshow(dobspad[ishot], aspect='auto', cmap='gray',
                              vmin=-d_vmax, vmax=d_vmax)
        else:
            wav = self.wav
            dobspad = dobs

        # Run inversion
        loss_hist = []
        for ifreq, freq in enumerate(self.frequencies):
            if freq is not None:
                print(f'Working with frequency {ifreq + 1}/{len(self.frequencies)}')

                # Filter wavelet
                wavfilt = Filt.apply_filter(wav.squeeze(), ifilt=ifreq)

                if plotflag:
                    WAVfilt = np.fft.rfft(wavfilt, self.nfft)

                    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
                    axs[0].plot(wav, 'k')
                    axs[0].plot(wavfilt, 'r')
                    axs[1].plot(f, np.abs(WAV), 'k')
                    axs[1].plot(f, np.abs(WAVfilt), 'r')
                    axs[1].set_xlim(0, freq * 2)

                # Filter data
                dobsfilt = np.vstack(
                    [Filt.apply_filter(dobspad[isrc].T, ifilt=ifreq).T[None, :] for isrc in range(self.par['ns'])])
            else:
                wavfilt = self.wav
                dobsfilt = dobspad

            if plotflag:
                # Plot shot gathers
                d_vmin, d_vmax = np.percentile(np.hstack(dobsfilt).ravel(), [2, 98])

                fig, axs = plt.subplots(1, 3, figsize=(14, 9))
                for ax, ishot in zip(axs, [0, self.par['ns'] // 2, self.par['ns'] - 1]):
                    ax.imshow(dobsfilt[ishot], aspect='auto', cmap='gray',
                              vmin=-d_vmax, vmax=d_vmax)

            # Create loss and wave engine
            lossfc = self.loss(self.lossop[ifreq], dobsfilt.reshape(self.par['ns'], -1))
            ainv = AcousticWave2D(self.shape, self.origin, self.spacing,
                                  self.x_s[:, 0], self.x_s[:, 1], self.x_r[:, 0], self.x_r[:, 1],
                                  0., self.tmax,
                                  vprange=self.vp_range,
                                  vpinit=self.vp_init,
                                  wav=wavfilt, f0=self.par['freq'],
                                  space_order=self.space_order, nbl=self.nbl,
                                  loss=lossfc)

            if self.firstscaling:
                # Compute first gradient and find scaling
                if self.postprocess is not None:
                    self.postprocess.scaling = 1.
                print('self.postprocess.scaling before _loss_grad', self.postprocess.scaling)
                loss, direction = ainv._loss_grad(ainv.initmodel.vp,
                                                  postprocess=None if self.postprocess is None else self.postprocess.apply)
                scaling = direction.max()
                print('Scaling', scaling)
                if self.postprocess is not None:
                    self.postprocess.scaling = scaling
                print('self.postprocess.scaling after _loss_grad', self.postprocess.scaling)
                if plotflag:
                    plt.figure(figsize=(14, 5))
                    im = plt.imshow(direction.T / scaling, cmap='seismic', vmin=-1e-1, vmax=1e-1,
                                    extent=(self.x[0], self.x[-1], self.z[-1], self.z[0]))
                    plt.scatter(self.x_r[:, 0], self.x_r[:, 1], c='w')
                    plt.scatter(self.x_s[:, 0], self.x_s[:, 1], c='r')
                    plt.title('Gradient')
                    plt.axis('tight')
                    plt.colorbar(im)

            # FWI with L-BFGS
            nl = minimize(ainv.loss_grad,
                          self.vp_init.ravel() * 1e-3, # km/s
                          method=self.solver, jac=True,
                          args=(None if self.convertvp is None else self.convertvp.apply,
                                None if self.postprocess is None else self.postprocess.apply),
                          callback=self.callback,
                          options=self.kwargs_solver)
            vp_inv = nl.x.reshape(self.shape)
            self.vp_init = vp_inv.copy() * 1e3 # m/s

            loss_hist.append(ainv.losshistory)

            if plotflag:
                if vlims is not None:
                    m_vmin, m_vmax = vlims
                else:
                    m_vmin, m_vmax = np.percentile(vp_inv, [2, 98])

                plt.figure(figsize=(14, 5))
                plt.imshow(vp_inv.T, vmin=m_vmin, vmax=m_vmax, cmap='jet', extent=(self.x[0], self.x[-1], self.z[-1], self.z[0]))
                plt.colorbar()
                plt.scatter(self.x_r[:, 0], self.x_r[:, 1], c='w')
                plt.scatter(self.x_s[:, 0], self.x_s[:, 1], c='r')
                plt.title('Inverted VP')
                plt.axis('tight')

        return vp_inv, loss_hist, nl