__all__ = ["AcousticWave2D"]

from typing import Any, Optional, NewType, Type, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt

from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike
from tqdm.autonotebook import tqdm

from examples.seismic import AcquisitionGeometry, Model
from examples.seismic.model import SeismicModel

from devitofwi.devito.acoustic.wavesolver import AcousticWaveSolver
from devitofwi.nonlinear import NonlinearOperator
from devitofwi.devito.source import CustomSource
from devitofwi.devito.utils import clear_devito_cache

try:
    from mpi4py import MPI
    mpitype = MPI.Comm
except:
    mpitype = Any

MPIType = NewType("MPIType", mpitype)


class AcousticWave2D(NonlinearOperator):
    """Devito Acoustic propagator.

    This class provides functionalities to model acoustic data and 
    perform full-waveform inversion with the Devito Acoustic propagator

    Parameters
    ----------
    shape : :obj:`tuple`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple`
        Model origin in km ``(ox, oz)``
    spacing : :obj:`tuple`
        Model spacing in km ``(dx, dz)``
    src_x : :obj:`numpy.ndarray`
        Source x-coordinates in km
    src_z : :obj:`numpy.ndarray` or :obj:`float`
        Source z-coordinates in km
    rec_x : :obj:`numpy.ndarray`
        Receiver x-coordinates in km
    rec_z : :obj:`numpy.ndarray` or :obj:`float`
        Receiver z-coordinates in km
    t0 : :obj:`float`
        Initial time in s
    tn : :obj:`float`
        Final time in s
    dt : :obj:`float`, optional
        Time step in s (if not provided this is directly inferred by devito)
    vp : :obj:`numpy.ndarray`, optional
        Velocity model in km/s for modelling of size :math:`n_x \times n_z`
        (use ``None`` if the data is already available)
    vprange : :obj:`tuple`, optional
        Velocity range in km/s ``(vmin, vmax)``, to be used in loss and gradient computations
        (can be provided instead of ``vp`` to create a propagator with a time axis 
        that is consistent with that of the data modelled with ``vp``)
    space_order : :obj:`int`, optional
        Spatial ordering of FD stencil
    nbl : :obj:`int`, optional
        Number ordering of samples in absorbing boundaries
    src_type : :obj:`str`, optional
        Source type
    f0 : :obj:`float`, optional
        Source peak frequency in Hz
    wav : :obj:`numpy.ndarray`, optional
        Wavelet (if provided ``src_type`` will be ignored)
    fs : :obj:'bool', optional
        Use free surface boundary at the top of the model.
    streamer_acquisition : :obj:'bool', optional
        Update receiver locations in geometry for each source
    checkpointing : :obj:`bool`, optional
        Use checkpointing (``True``) or not (``False``). Note that
        using checkpointing is needed when dealing with large models.
        Cannot be used with snapshotting (factor).
    factor : :obj:`int`, optional 
        Subsampling factor to use snapshots of the wavefield to compute the gradient.
        Cannot be used with checkpointing.
    loss : :obj:`Type`, optional
        Loss object.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    clearcache : :obj:`bool`, optional
        Clear devito cache (``True``) or not (``False``) after every modelling step
    base_comm : :obj:`mpi4py.MPI.Comm`, optional
        Base MPI Communicator. Defaults to ``mpi4py.MPI.COMM_WORLD``.
    sub_gradient : :obj:'bool', optional
        If True, restricts the computation domain for each shot gather to a specified portion of the model.
        By default, the domain spans the maximum offset of the data plus an additional 1 km on 
        both the left and right sides.
    extent : :obj:`tuple`
        A tuple specifying the extent (in km) to extend the computation domain on the left and right for 
        sub_gradient. Default is (1.0, 1.0) km. This parameter is only used when sub_gradient is True.
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
        vprange: Optional[Tuple] = None,
        space_order: Optional[int] = 4,
        nbl: Optional[int] = 20,
        src_type: Optional[str] = "Ricker",
        f0: Optional[float] = 20.0,
        wav: Optional[NDArray] = None,
        fs: Optional[bool] = False,
        streamer_acquisition: Optional[bool] = False,
        checkpointing: Optional[bool] = False,
        factor: Optional[int] = None,
        loss: Optional[Type] = None,
        dtype: Optional[DTypeLike] = "float32",
        clearcache: Optional[bool] = False,
        base_comm: Optional[MPIType] = None,
        sub_gradient: Optional[bool] = False,
        extent: Optional[Tuple] = (1., 1.),
    ) -> None:

        # Check to ensure that vp or vprange is provided
        if vp is None and vprange is None:
            raise ValueError("Provide either vp or vprange, not none...")
        elif vp is not None and vprange is not None:
            raise ValueError("Provide either vp or vprange, not both...")

        # Create vp if not provided and vprange is available
        if vprange is not None:
            vp = vprange[0] * np.ones(shape)
            vp[:, -1] = vprange[1]
        
        # Geometry parameters
        self.src = (src_x, src_z)
        self.rec = (rec_x, rec_z)

        # Modelling parameters
        self.shape = shape
        self.origin = origin
        self.spacing = spacing
        self.t0 = t0
        self.tn = tn
        self.dt = dt
        self.space_order = space_order
        self.nbl = nbl
        self.src_type = src_type
        self.f0 = f0
        self.wav = wav
        self.fs = fs
        self.streamer_acquisition = streamer_acquisition
        self.checkpointing = checkpointing
        self.factor = factor
        self.clearcache = clearcache
        self.sub_gradient = sub_gradient
        self.extent = extent
        

        # Store model
        self.vp = vp

        # Inversion parameters
        self.loss = loss
        self.losshistory = []

        # MPI parameters
        self.base_comm = base_comm
    
        super().__init__(size=np.prod(shape), dtype=dtype)


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
            Model origin in km ``(ox, oz)``
        spacing : :obj:`numpy.ndarray`
            Model spacing in km ``(dx, dz)``
        vp : :obj:`numpy.ndarray`
            Velocity model in km/s
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
            vp=vp,
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
            Source x-coordinates in km
        src_z : :obj:`numpy.ndarray` or :obj:`float`
            Source z-coordinates in km
        rec_x : :obj:`numpy.ndarray`
            Receiver x-coordinates in km
        rec_z : :obj:`numpy.ndarray` or :obj:`float`
            Receiver z-coordinates in km
        t0 : :obj:`float`
            Initial time in s
        tn : :obj:`float`
            Final time in s
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency in Hz
        dt : :obj:`float`, optional
            Time step time in s (if provided, the geometry time_axis is
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
            f0=None if f0 is None else f0,
            fs=self.fs,
        )

        # Resample geometry to user defined dt
        if dt is not None:
            geometry.resample(dt)

        return geometry

    def model_and_geometry(self):
        model = self._create_model(self.shape, self.origin, self.spacing, 
                                   self.vp, self.space_order, self.nbl, self.fs)
        geometry = self._create_geometry(model,
                                         self.src[0][:1], self.src[1][:1], self.rec[0], self.rec[1], 
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
        return model, geometry

    def _get_location(self, isrc: int):
        # Calculate maximum offset in grid units
        max_offset = (self.rec[0][-1] - self.rec[0][0])

        # Compute x0 with grid conversion and boundary checking
        x0 = max(0, math.floor((self.src[0][isrc] - self.extent[0]) / self.spacing[0]))
        
        # Compute xf with grid conversion and boundary checking
        xf = min(math.ceil((self.src[0][isrc] + self.extent[1] + max_offset) / self.spacing[0]), self.shape[0] - 1)
    
        return (x0, xf)
    
    def _mod_oneshot(self, model: SeismicModel, isrc: int, dt: float = None) -> NDArray:
        """FD modelling for one shot

        Parameters
        ----------
        model : :obj:`examples.seismic.model.SeismicModel`
            Model
        isrc : :obj:`int`
            Index of source to model
        dt : :obj:`float`, optional
            Time sampling in s used to resample modelled data

        Returns
        -------
        d : :obj:`np.ndarray`
            Data of size ``nr \times nt``
        dt : :obj:`float`, optional
            Time sampling in s of modelled data
        
        """
        # Create geometry
        geometry = self._create_geometry(model,
                                         self.src[0][:1], self.src[1][:1], self.rec[0], self.rec[1], 
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
        
        # Update source location in geometry
        geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc])
        if self.streamer_acquisition:
            # Update receiver locations in geometry
            geometry.rec_positions[:, 0] = geometry.src_positions[0, 0] + self.rec[0]
        
        # Re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=model.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)
            geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc])
            src.coordinates.data[0, :] = (self.src[0][isrc], self.src[1][isrc])

        # Solve
        solver = AcousticWaveSolver(model, geometry, 
                                    space_order=self.space_order)
        d, _, _, _ = solver.forward(vp=model.vp, src=src, autotune=True)

        # Resample
        if dt is None:
            dt = geometry.dt
            d = d.data.copy()
        else:
            d = d.resample(dt).data.copy()
        
        return d, dt

    def mod_allshots(self, dt=None) -> NDArray:
        """FD modelling for all shots

        Parameters
        ----------
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data in s

        Returns
        -------
        dtot : :obj:`np.ndarray`
            Data for all shots
        dt : :obj:`float`, optional
            Time sampling in s of modelled data
        
        """
        # Create model
        if not self.sub_gradient:
            model = self._create_model(self.shape, self.origin, self.spacing, 
                                    self.vp, self.space_order, self.nbl, self.fs)

        # Run modelling
        nsrc = self.src[0].size
        dtot = []
        for isrc in tqdm(range(nsrc)):
            if self.sub_gradient:
                x0, xf = self._get_location(isrc)
    
                model = self._create_model((xf-x0, self.shape[1]), (x0*self.spacing[0], self.origin[1]), self.spacing, 
                                        self.vp[x0:xf], self.space_order, self.nbl, self.fs)
            d, dt = self._mod_oneshot(model, isrc, dt)
            if isrc == 0:
                nt_max = d.shape[0]
            elif d.shape[0] < nt_max:
                nt_max = d.shape[0]
            dtot.append(d)
            if self.clearcache:
                clear_devito_cache()
        dtot = np.array([d[:nt_max] for d in dtot]).reshape(nsrc, nt_max, d.shape[1])
        
        return dtot, dt

    def mod_allshots_mpi(self, dt=None) -> NDArray:
        """FD modelling for all shots with mpi gathering

        Parameters
        ----------
        dt : :obj:`float`, optional
            Time sampling used to resample modelled data in s

        Returns
        -------
        d : :obj:`np.ndarray`
            Data for all shots
        dt : :obj:`float`, optional
            Time sampling in s of modelled data
        
        """
        dtotrank, dt = self.mod_allshots(dt)

        # gather shots from all ranks
        dtot = np.concatenate(self.base_comm.allgather(dtotrank), axis=0)
        
        return dtot, dt

    def _adjoint_source(self, d_syn, isrc):
        """Adjoint source computation

        Note to self, takes flatten inputs and returns flatten outputs
        """
        return self.loss.grad(d_syn, isrc)

    def _loss_grad_oneshot(self, vp, src, solver, isrc,
                           computeloss=True, computegrad=True) -> Tuple[float, NDArray]:
        """Raw loss function and gradient for one shot

        Compute raw loss function and gradient for one shot without applying any pre/post-processing. Note
        that Devito returns the gradient for slowness square.

        """
        # Compute synthetic data and full forward wavefield u0
        adjsrc, u0, usnaps, _ = solver.forward(vp=vp, save=True if self.factor is None else False,
                                               src=src, autotune=True, factor=self.factor)
        
        # Compute loss
        if computeloss:
            loss = self.loss(adjsrc.data[:].ravel(), isrc)
        if computegrad:
            # Compute adjoint source
            adjsrc.data[:] = self._adjoint_source(adjsrc.data[:].ravel(), isrc).reshape(adjsrc.data.shape)

            # Compute gradient
            grad, _ = solver.gradient(rec=adjsrc, u=u0, usnaps=usnaps, vp=vp, checkpointing=self.checkpointing, autotune=True,
                                      factor=self.factor)

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
        vp : :obj:`numpy.ndarray`
            Velocity model in km/s
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
            nsrc = self.src[0].size
            isrcs = range(nsrc)
        
        # Create model with class vp to define a geometry and time axis consistent with 
        # the observed data and one with provided vp (to be used as input for loss and
        # gradient computation)
        if not self.sub_gradient:
            model = self._create_model(self.shape, self.origin, self.spacing, 
                                    self.vp, self.space_order, self.nbl, self.fs)
            modelvp = self._create_model(self.shape, self.origin, self.spacing, 
                                        vp, self.space_order, self.nbl, self.fs)
            
            

            # Geometry for single source
            geometry = self._create_geometry(model,
                                            self.src[0][:1], self.src[1][:1], self.rec[0], self.rec[1], 
                                            self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)

            # Re-create source (if wav is not None)
            if self.wav is None:
                src = geometry.src
            else:
                src = CustomSource(name='src', grid=model.grid,
                                wav=self.wav, npoint=1,
                                time_range=geometry.time_axis)

            # Solver
            solver = AcousticWaveSolver(model, geometry,
                                        space_order=self.space_order)
        
        # Compute loss and gradient
        loss = 0.
        for isrc in tqdm(isrcs):
            if self.sub_gradient:
                x0, xf = self._get_location(isrc)

                model = self._create_model((xf-x0, self.shape[1]), (x0*self.spacing[0], self.origin[1]), self.spacing, 
                                        self.vp[x0:xf], self.space_order, self.nbl, self.fs)
                
                modelvp = self._create_model((xf-x0, self.shape[1]), (x0*self.spacing[0], self.origin[1]), self.spacing, 
                                            vp[x0:xf], self.space_order, self.nbl, self.fs)
                geometry = self._create_geometry(model,
                                         self.src[0][isrc:isrc+1], self.src[1][:1], self.rec[0], self.rec[1], 
                                         self.t0, self.tn, self.src_type, f0=self.f0, dt=self.dt)
                # Re-create source (if wav is not None)
                if self.wav is None:
                    src = geometry.src
                else:
                    src = CustomSource(name='src', grid=model.grid,
                                    wav=self.wav, npoint=1,
                                    time_range=geometry.time_axis)
                solver = AcousticWaveSolver(model, geometry,
                                    space_order=self.space_order)
            # Update source location in geometry
            geometry.src_positions[0, :] = (self.src[0][isrc], self.src[1][isrc])
            src.coordinates.data[0, :] = (self.src[0][isrc], self.src[1][isrc])
            if self.streamer_acquisition:
                # Update receiver locations in geometry
                geometry.rec_positions[:, 0] = geometry.src_positions[0, 0] + self.rec[0]
            
            # Compute loss and gradient for one shot
            lossgrad = self._loss_grad_oneshot(modelvp.vp, src, solver, isrc,
                                               computeloss=computeloss, 
                                               computegrad=computegrad)
            if computeloss and computegrad:
                loss += lossgrad[0]
                if isrc == 0:
                    if self.sub_gradient:
                        grad = self._crop_model(lossgrad[1].data[:], self.nbl, self.fs)
                        full_grad = np.zeros(self.shape, dtype=np.float64)
                        full_grad[x0:xf] = grad.copy()
                    else:
                        grad = lossgrad[1].data[:]
                else:
                    if self.sub_gradient:
                        grad = self._crop_model(lossgrad[1].data[:], self.nbl, self.fs)
                        full_grad[x0:xf] += grad.copy()
                    else:
                        grad += lossgrad[1].data[:]
            elif computeloss:
                loss += lossgrad
            elif computegrad:
                if isrc == 0:
                    if self.sub_gradient:
                        grad = self._crop_model(lossgrad.data[:], self.nbl, self.fs)
                        full_grad = np.zeros(self.shape, dtype=np.float64)
                        full_grad[x0:xf] = grad.copy()
                    else:
                        grad = lossgrad.data[:]
                else:
                    if self.sub_gradient:
                        grad = self._crop_model(lossgrad.data[:], self.nbl, self.fs)
                        full_grad[x0:xf] += grad.copy()
                    else:
                        grad += lossgrad.data[:]
            
        if self.clearcache:
                clear_devito_cache()

        # Gather gradients
        if self.base_comm is not None:
            if computeloss:
                loss = self.base_comm.allreduce(loss, op=MPI.SUM)
            if computegrad:
                grad = self.base_comm.allreduce(grad, op=MPI.SUM)

        # Postprocess loss and gradient
        
        grad = self._crop_model(grad, self.nbl, self.fs) if not self.sub_gradient else full_grad
        if self.sub_gradient:
            modelvp_ = self._create_model(self.shape, self.origin, self.spacing, 
                                        vp, self.space_order, self.nbl, self.fs)
            vp = self._crop_model(modelvp_.vp.data[:], self.nbl, self.fs)
        else:
            vp = self._crop_model(modelvp.vp.data[:], self.nbl, self.fs)
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
            vp = x.reshape(self.shape)
        else:
            vp = convertvp(x.reshape(self.shape))

        # Evaluate objective function and gradient
        lossgrad = self._loss_grad(vp.reshape(self.shape),
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
            print('Debug - loss, grad.min(), grad.max()',
                  loss, grad.min(), grad.max())
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
