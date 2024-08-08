__all__ = ["AcousticWave2D"]

from typing import Any, Optional, NewType, Type, Tuple

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike
from tqdm.autonotebook import tqdm

from devito import Function
from examples.seismic import AcquisitionGeometry, Model, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from devitofwi.devito.source import CustomSource

import matplotlib.pyplot as plt

try:
    from mpi4py import MPI
    mpitype = MPI.Comm
except:
    mpitype = Any

MPIType = NewType("MPIType", mpitype)


class AcousticWave2D():
    """Devito Acoustic propagator.

    This class provides functionalities to model acoustic data and 
    perform full-waveform inversion with the Devito Acoustic propagator

    Parameters
    ----------
    shape : :obj:`tuple`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple`
        Model spacing ``(dx, dz)``
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in m
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in m
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in m
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in m
    t0 : :obj:`float`
        Initial time in ms
    tn : :obj:`int`
        Final time in ms
    dt : :obj:`float`, optional
        Time step in ms (if not provided this is directly inferred by devito)
    vp : :obj:`numpy.ndarray`, optional
        Velocity model in m/s for modelling, 
        (use ``None`` if the data is already available)
    vpinit : :obj:`numpy.ndarray`, optional
        Initial velocity model in m/s as starting guess for inversion
    vprange : :obj:`tuple`, optional
        Velocity range (min, max) to be used in loss and gradient computations
        (can be provided instead of ``vp`` to create a propagator for ``vpinit``
        with a time axis that is however consistent with that of the data modelled with ``vp``)
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    src_type : :obj:`str`, optional
        Source type
    f0 : :obj:`float`, optional
        Source peak frequency (Hz)
    wav : :obj:`numpy.ndarray`, optional
        Wavelet (if provided ``src_type`` and ``f0`` will be ignored
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models
    loss : :obj:`Type`, optional
        Loss object.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        Base MPI Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    fs : :obj:'bool', optional
        Use free surface boundary at the top of the model.
    streamer_acquisition : :obj:'bool', optional
        Update receiver locations in geometry for each source

    """

    def __init__(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        dt: Optional[float] = None,
        vp: Optional[NDArray] = None,
        vpinit: Optional[NDArray] = None,
        vprange: Optional[Tuple] = None,
        space_order: Optional[int] = 4,
        nbl: Optional[int] = 20,
        src_type: Optional[str] = "Ricker",
        f0: Optional[float] = 20.0,
        wav: Optional[NDArray] = None,
        checkpointing: Optional[bool] = False,
        loss: Optional[Type] = None,
        dtype: Optional[DTypeLike] = "float32",
        base_comm: Optional[MPIType] = None,
        fs: Optional[bool] = False,
        streamer_acquisition: Optional[bool] = False,
    ) -> None:

        # Create vp if not provided and vprange is available
        if vp is None and vprange is not None:
            vp = vprange[0] * np.ones(shape)
            vp[-1, -1] = vprange[1]

        # Velocity checks to ensure either vp or vint are provided
        if vp is None and vpinit is None:
            raise ValueError("Either vp or vpinit must be provided...")
        #if vpinit is not None and loss is None:
        #    raise ValueError("Must provide a loss to be able to run inversion...")

        # Modelling parameters
        self.space_order = space_order
        self.nbl = nbl
        self.fs = fs
        self.streamer_acquisition = streamer_acquisition
        self.checkpointing = checkpointing
        self.wav = wav

        # Inversion parameters
        self.loss = loss
        self.losshistory = []

        # MPI parameters
        self.base_comm = base_comm
        
        # Create model
        self.modelexists = True if vp is not None else False

        if vpinit is not None:
            self.initmodel = self._create_model(shape, origin, spacing, vpinit, space_order, nbl, fs)
        if vp is not None:
            self.model = self._create_model(shape, origin, spacing, vp, space_order, nbl, fs)
        # else:
        #    self.model = self._create_model(shape, origin, spacing, vpinit, space_order, nbl, fs)

        # Create geometry
        self.geometry = self._create_geometry(self.model if vp is not None else self.initmodel,
                                              src_x, src_z, rec_x, rec_z, t0, tn, src_type,
                                              f0=f0, dt=dt)
        self.geometry1shot = self._create_geometry(self.model if vp is not None else self.initmodel,
                                                   src_x[:1], src_z[:1], rec_x, rec_z, t0, tn, src_type,
                                                   f0=f0, dt=dt)

    @staticmethod
    def _crop_model(m: NDArray, nbl: int, fs: bool) -> NDArray:
        """Remove absorbing boundaries from model"""
        if fs:
            return m[nbl:-nbl, :-nbl]
        else:
            return m[nbl:-nbl, nbl:-nbl]

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        space_order: int = 4,
        nbl: int = 20,
        fs: bool = False,
    ) -> None:
        """Create model

        Parameters
        ----------
        shape : :obj:`numpy.ndarray`
            Model shape ``(nx, nz)``
        origin : :obj:`numpy.ndarray`
            Model origin ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing ``(dx, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in m/s
        space_order : :obj:`int`, optional
            Spatial ordering of FD stencil
        nbl : :obj:`int`, optional
            Number ordering of samples in absorbing boundaries
        fs : :obj:'bool', optional
            Use free surface boundary at the top of the model.

        Returns
        -------
        model : :obj:`examples.seismic.model.SeismicModel`
            Model
        
        """
        model = Model(
            space_order=space_order,
            vp=vp * 1e-3,
            origin=origin,
            shape=shape,
            dtype=np.float32,
            spacing=spacing,
            nbl=nbl,
            bcs="damp",
            fs=fs,
        )
        return model

    def _create_geometry(
        self,
        model,
        src_x: NDArray,
        src_z: NDArray,
        rec_x: NDArray,
        rec_z: NDArray,
        t0: float,
        tn: float,
        src_type: str,
        f0: float = 20.0,
        dt: float = None
    ) -> None:
        """Create geometry and time axis

        Parameters
        ----------
        model : :obj:`examples.seismic.model.SeismicModel`
            Model
        src_x : :obj:`numpy.ndarray`
            Source x-coordinates in m
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in m
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in m
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in m
        t0 : :obj:`float`
            Initial time in ms
        tn : :obj:`float`
            Final time in ms
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)
        dt : :obj:`float`, optional
            Time step time in ms (if provided, the geometry time_axis is
            recreated with this time step)

        """
        nsrc, nrec = len(src_x), len(rec_x)
        src_coordinates = np.empty((nsrc, 2))
        src_coordinates[:, 0] = src_x
        src_coordinates[:, 1] = src_z

        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_z

        geometry = AcquisitionGeometry(
            model,
            rec_coordinates,
            src_coordinates,
            t0,
            tn,
            src_type=src_type,
            f0=None if f0 is None else f0 * 1e-3,
            fs=self.model.fs,
        )

        # Resample geometry to user defined dt
        if dt is not None:
            geometry.resample(dt)

        return geometry

    def _mod_oneshot(self, isrc: int, dt: float = None) -> NDArray:
        """FD modelling for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data

        Returns
        -------
        d : :obj:`np.ndarray`
            Data of size ``nr \times nt``

        """
        # Update source location in geometry
        geometry = self.geometry1shot
        geometry.src_positions[0, :] = self.geometry.src_positions[isrc, :]
        if self.streamer_acquisition:
            # Update receiver locations in geometry
            geometry.rec_positions[:, 0] = geometry.src_positions[0, 0] + geometry.rec_positions[:, 0]

        # Re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=self.model.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)
            src.coordinates.data[0, :] = self.geometry.src_positions[isrc, :]

        # Create data object
        d = Receiver(name='data', grid=self.model.grid,
                     time_range=geometry.time_axis, 
                     coordinates=geometry.rec_positions)
        # Solve
        solver = AcousticWaveSolver(self.model, geometry, 
                                    space_order=self.space_order)
        _, _, _ = solver.forward(vp=self.model.vp, rec=d, src=src)

        # Resample
        if dt is None:
            d = d.data.copy()
        else:
            d = d.resample(dt).data.copy()
        return d

    def mod_allshots(self, dt=None) -> NDArray:
        """FD modelling for all shots

        Parameters
        ----------
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots

        """
        nsrc = self.geometry.src_positions.shape[0]
        dtot = []

        for isrc in tqdm(range(nsrc)):
            d = self._mod_oneshot(isrc, dt)
            dtot.append(d)
        dtot = np.array(dtot).reshape(nsrc, d.shape[0], d.shape[1])
        
        return dtot

    def mod_allshots_mpi(self, dt=None) -> NDArray:
        """FD modelling for all shots with mpi gathering

        Parameters
        ----------
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data

        Returns
        -------
        d : :obj:`np.ndarray`
            Data for all shots

        """
        dtotrank = self.mod_allshots(dt)

        # gather shots from all ranks
        dtot = np.concatenate(self.base_comm.allgather(dtotrank), axis=0)
        
        return dtot

    def _adjoint_source(self, d_syn, isrc):
        """Adjoint source computation

        Note to self, takes flatten inputs and returns flatten outputs
        """
        return self.loss.grad(d_syn, isrc)

    def _loss_grad_oneshot(self, vp, src, solver, d_syn, adjsrc, grad, isrc,
                           computeloss=True, computegrad=True) -> Tuple[float, NDArray]:
        """Raw loss function and gradient for one shot

        Compute raw loss function and gradient for one shot without applying any pre/post-processing. Note
        that Devito returns the gradient for slowness square.

        """
        # Compute synthetic data and full forward wavefield u0
        _, u0, _ = solver.forward(vp=vp, save=True, rec=d_syn, src=src)
        
        # Compute loss
        if computeloss:
            loss = self.loss(d_syn.data[:].ravel(), isrc)
        if computegrad:
            # Compute adjoint source
            adjsrc.data[:] = self._adjoint_source(d_syn.data[:].ravel(), isrc).reshape(adjsrc.data.shape)

            # Compute gradient
            solver.gradient(rec=adjsrc, u=u0, vp=vp, grad=grad, checkpointing=self.checkpointing)
        
        if computeloss and computegrad:
            return loss, grad
        elif computeloss:
            return loss
        else:
            return grad 
        
    def _loss_grad(self, vp, isrcs=None, postprocess=None, computeloss=True, computegrad=True):
        """Compute loss function and gradient
        
        Parameters
        ----------
        vp : :obj:`devito.Function`
            Velocity model
        isrcs : :obj:`list`, optional
            Indices of shots to be used in gradient computation 
            (if ``None``, use all shots whose number is inferred from ``dobs``)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss
        computeloss : :obj:`bool`, optional
            Compute loss function
        computegrad : :obj:`bool`, optional
            Compute gradient

        Returns
        -------
        loss : :obj:`float`
            Loss function
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, nz)``

        """
        # Identify number of shots
        if isrcs is None:
            nsrc = self.geometry.src_positions.shape[0]
            isrcs = range(nsrc)

        # Geometry for single source
        geometry = self.geometry1shot

        # Re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=self.model.grid if self.modelexists else self.initmodel.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)

        # Solver
        solver = AcousticWaveSolver(self.model if self.modelexists else self.initmodel,
                                    geometry,
                                    space_order=self.space_order)
        
        # Symbols to hold the observed data, modelled data, adjoint source, and gradient
        d_syn = Receiver(name='d_syn', grid=self.initmodel.grid,
                         time_range=geometry.time_axis, 
                         coordinates=geometry.rec_positions)
        adjsrc = Receiver(name='adjsrc', grid=self.initmodel.grid,
                          time_range=geometry.time_axis, 
                          coordinates=geometry.rec_positions)
        grad = Function(name="grad", grid=self.initmodel.grid)

        # Compute loss and gradient
        loss = 0.
        for isrc in tqdm(isrcs):
            # Update source location in geometry
            geometry.src_positions[0, :] = self.geometry.src_positions[isrc, :]
            src.coordinates.data[0, :] = self.geometry.src_positions[isrc, :]
            if self.streamer_acquisition:
                # Update receiver locations in geometry
                geometry.rec_positions[:, 0] = geometry.src_positions[0, 0] + geometry.rec_positions[:, 0]
                
                d_syn = Receiver(name='d_syn', grid=self.initmodel.grid,
                                 time_range=geometry.time_axis, 
                                 coordinates=geometry.rec_positions)
                adjsrc = Receiver(name='adjsrc', grid=self.initmodel.grid,
                                  time_range=geometry.time_axis, 
                                  coordinates=geometry.rec_positions)
            # Compute loss and gradient for one shot
            lossgrad = self._loss_grad_oneshot(vp, src, solver, d_syn, adjsrc, grad, isrc,
                                               computeloss=computeloss, computegrad=computegrad)
            if computeloss and computegrad:
                loss += lossgrad[0]
            elif computeloss:
                loss += lossgrad

        if computegrad:
            grad = grad.data[:]

        # Gather gradients 
        if self.base_comm is not None:
            if computeloss:
                loss = self.base_comm.allreduce(loss, op=MPI.SUM)
            if computegrad:
                grad = self.base_comm.allreduce(grad, op=MPI.SUM)

        # Postprocess loss and gradient
        grad = self._crop_model(grad, self.nbl, self.fs)
        vp = self._crop_model(vp.data[:], self.nbl, self.fs)
        if postprocess is not None:
            loss, grad = postprocess(vp, loss, grad)

        if computeloss and computegrad:
            return loss, grad
        elif computeloss:
            return loss
        else:
            return grad 
        
    def loss_grad(self, x, convertvp=None, postprocess=None,
                  computeloss=True, computegrad=True,
                  debug=False, gradlims=None):
        """Compute loss function and gradient to be used by solver

        This routine wraps _loss_grad providing and returning numpy arrays 
        and should be used with any solver
        
        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertvp : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss
        computeloss : :obj:`bool`, optional
            Compute loss function
        computegrad : :obj:`bool`, optional
            Compute gradient
        debug : :obj:`bool`, optional
            Debugging flag
        gradlims : :obj:`tuple`, optional
            Limits of gradient to be used in plotting when ``debug=True``

        Returns
        -------
        loss : :obj:`float`
            Loss function
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, nz)``

        """

        # Convert x to velocity
        if convertvp is None:
            vp = x.reshape(self.initmodel.shape)
        else:
            vp = convertvp(x.reshape(self.initmodel.shape))

        # Overwrite current velocity in devito model used to compute the synthetic data
        self.initmodel.update('vp', vp.reshape(self.initmodel.shape))
        
        # Evaluate objective function and gradient
        lossgrad = self._loss_grad(self.initmodel.vp,
                                   postprocess=postprocess,
                                   computeloss=computeloss,
                                   computegrad=computegrad)

        # Split lossgrad based on what has been computed in self._loss_grad
        if computeloss and computegrad:
            loss, grad = lossgrad
        elif computeloss:
            loss, grad = lossgrad, None
        else:
            loss, grad = None, lossgrad

        # Save loss history
        if computeloss:
            self.losshistory.append(loss)
        
        # Display results in debugging mode
        if debug and computeloss and computegrad:
            print('Debug - loss, scaling, grad.min(), grad.max()',
                  fval, scaling, grad.min(), grad.max())
            plt.figure()
            plt.imshow(grad.T, vmin=gradlims[0] if gradlims is not None else -grad.max(),
                       vmax=gradlims[1] if gradlims is not None else grad.max(),
                       aspect='auto', cmap='seismic')
            plt.colorbar()

        # Return loss, grad or both
        if computeloss and computegrad:
            return loss, grad.ravel()
        elif computeloss:
            return loss
        else:
            return grad.ravel()

    def loss(self, x, convertvp=None, postprocess=None):
        """Compute loss function to be used by solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertvp : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss

        Returns
        -------
        loss : :obj:`float`
            Loss function

        """
        return self.loss_grad(x, convertvp=convertvp, postprocess=postprocess,
                              computeloss=True, computegrad=False)

    def grad(self, x, convertvp=None, postprocess=None):
        """Compute gradient to be used by solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Model obtained by the solver
        convertvp : :obj:`func`, optional
            Function handle that converts the model obtained by the solver in velocity to be used by the propagator
            (if ``None``, it is assumed that the solver itself is working with a velocity model)
        postprocess : :obj:`funct`, optional
            Function handle applying postprocessing to gradient and loss

        Returns
        -------
        grad : :obj:`numpy.ndarray`
            Gradient of size ``(nx, nz)``

        """
        return self.loss_grad(x, convertvp=convertvp, postprocess=postprocess,
                              computeloss=False, computegrad=True)
