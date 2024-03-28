__all__ = ["AcousticWave2D"]

from typing import Optional, Type, Tuple

import numpy as np

from pylops import LinearOperator
from pylops.utils import deps
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray, SamplingLike
from tqdm.notebook import tqdm


from devito import Function
from examples.seismic import AcquisitionGeometry, Model, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from devitofwi.source import CustomSource

#class AcousticWave2D(LinearOperator):
class AcousticWave2D():
    """Devito Acoustic propagator.

    This class provides functionalities to model acoustic data and 
    to perform full-waveform inversion with the Devito Acoustic propagator 

    Parameters
    ----------
    shape : :obj:`tuple` or :obj:`numpy.ndarray`
        Model shape ``(nx, nz)``
    origin : :obj:`tuple` or :obj:`numpy.ndarray`
        Model origin ``(ox, oz)``
    spacing : :obj:`tuple` or  :obj:`numpy.ndarray`
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
        Initial time in s
    tn : :obj:`int`
        Final time in s
    vp : :obj:`numpy.ndarray`, optional
        Velocity model in m/s for modelling, 
        (use``None`` if the data is already available)
    vpinit : :obj:`numpy.ndarray`, optional
        Initial velocity model in m/s as starting guess for inversion
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
        tn: int,
        vp: Optional[NDArray] = None,
        vpinit: Optional[NDArray] = None,
        space_order: Optional[int] = 4,
        nbl: Optional[int] = 20,
        src_type: Optional[str] = "Ricker",
        f0: Optional[float] = 20.0,
        wav: Optional[NDArray] = None,
        checkpointing: Optional[bool] = False,
        loss: Optional[Type] = None,
        dtype: Optional[DTypeLike] = "float32",
    ) -> None:
        # velocity checks to ensure either vp or vint are provided
        if vp is None and vpinit is None:
            raise ValueError("Either vp or vpinit must be provided...")
        if vpinit is not None and loss is None:
            raise ValueError("Must provide a loss to be able to run inversion...")
        
        # modelling parameters
        self.space_order = space_order
        self.nbl = nbl
        self.checkpointing = checkpointing
        self.wav = wav

        # inversion parameters
        self.loss = loss
        self.losshistory = []

        # create model
        if vp is not None:
            self.model = self._create_model(shape, origin, spacing, vp, space_order, nbl)
        if vpinit is not None:
            self.initmodel = self._create_model(shape, origin, spacing, vpinit, space_order, nbl)

        # create geometry
        self.geometry = self._create_geometry(self.model if vp is not None else self.initmodel, 
                                              src_x, src_z, rec_x, rec_z, t0, tn, src_type, f0=f0)
        self.geometry1shot = self._create_geometry(self.model if vp is not None else self.initmodel, 
                                                   src_x[:1], src_z[:1], rec_x, rec_z, t0, tn, src_type, f0=f0)

        # MUST BE CHANGED WHEN CREATING NONLINEAR OPERATOR
        #super().__init__(
        #    dtype=np.dtype(dtype),
        #    dims=vp.shape,
        #    dimsd=(len(src_x), len(rec_x), self.geometry.nt),
        #    explicit=False,
        #)

    @staticmethod
    def _crop_model(m: NDArray, nbl: int) -> NDArray:
        """Remove absorbing boundaries from model"""
        return m[nbl:-nbl, nbl:-nbl]

    def _create_model(
        self,
        shape: InputDimsLike,
        origin: SamplingLike,
        spacing: SamplingLike,
        vp: NDArray,
        space_order: int = 4,
        nbl: int = 20,
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
        tn: int,
        src_type: str,
        f0: float = 20.0,
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
            Initial time
        tn : :obj:`int`
            Final time in s
        src_type : :obj:`str`
            Source type
        f0 : :obj:`float`, optional
            Source peak frequency (Hz)

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
        )
        return geometry

    def _mod_oneshot(self, isrc: int, dt: float = None) -> NDArray:
        """FD modelling for one shot

        Parameters
        ----------
        isrc : :obj:`int`
            Index of source to model
        
        Returns
        -------
        d : :obj:`np.ndarray`
            Data

        """
        # update source location in geometry
        geometry = self.geometry1shot
        geometry.src_positions[0, :] = self.geometry.src_positions[isrc, :]

        # re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=self.model.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)
            src.coordinates.data[0, :] = self.geometry.src_positions[isrc, :]

        # data object
        d = Receiver(name='data', grid=self.model.grid,
                     time_range=geometry.time_axis, 
                     coordinates=geometry.rec_positions)
        # solve
        solver = AcousticWaveSolver(self.model, geometry, 
                                    space_order=self.space_order)
        _, _, _ = solver.forward(vp=self.model.vp, rec=d, src=src)

        # resample
        if dt is None:
            d = d.data.copy()
        else:
            d = d.resample(dt).data.copy()
        return d

    def _mod_allshots(self, dt=None) -> NDArray:
        """FD modelling for all shots

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

    def _loss_grad_oneshot(self, vp, geometry, src, solver, d_obs, d_syn, adjsrc, grad, dobs, isrc,
                           computeloss=True, computegrad=True) -> Tuple[float, NDArray]:
        
        # Generate synthetic data from true model
        #d_obs.data[:] = dobs
        
        # Compute smooth data and full forward wavefield u0
        _, u0, _ = solver.forward(vp=vp, save=True, rec=d_syn, src=src)
        
        # Compute loss
        if computeloss:
            #lossgrad = d_syn.data[:] - d_obs.data[:]
            #loss = .5 * np.linalg.norm(lossgrad) ** 2
            loss = self.loss(d_syn.data[:].ravel(), isrc)
        if computegrad:
            # Compute adjoint source
            #adjsrc.data[:] = lossgrad
            adjsrc.data[:] = self.loss.grad(d_syn.data[:].ravel(), isrc).reshape(adjsrc.data.shape)
            
            # Compute gradient
            solver.gradient(rec=adjsrc, u=u0, vp=vp, grad=grad)
        
        if computeloss and computegrad:
            return loss, grad
        elif computeloss:
            return loss
        else:
            return grad 
        
    def _loss_grad(self, vp, dobs, isrcs=None, mask=None, computeloss=True, computegrad=True):
        """Compute loss function and gradient
        
        Parameters
        ----------
        vp : :obj:`devito.Function`
            Velocity model
        dobs : :obj:`np.ndarray`
            Observed data for all shots
        isrcs : :obj:`list`, optional
            Indices of shots to be used in gradient computation 
            (if ``None``, use all shots whose number is inferred from ``dobs``)
        mask : :obj:`np.ndarray`, optional
            Mask to apply to gradient

        """
        # geometry for single source
        geometry = self.geometry1shot

        # re-create source (if wav is not None)
        if self.wav is None:
            src = geometry.src
        else:
            src = CustomSource(name='src', grid=self.model.grid,
                               wav=self.wav, npoint=1,
                               time_range=geometry.time_axis)

        # solver
        solver = AcousticWaveSolver(self.model, geometry, 
                                    space_order=self.space_order)
        
        # symbols to hold the observed data, modelled data, adjoint source, and gradient
        d_obs = Receiver(name='d_obs', grid=self.initmodel.grid,
                         time_range=geometry.time_axis, 
                         coordinates=geometry.rec_positions)
        d_syn = Receiver(name='d_syn', grid=self.initmodel.grid,
                         time_range=geometry.time_axis, 
                         coordinates=geometry.rec_positions)
        adjsrc = Receiver(name='adjsrc', grid=self.initmodel.grid,
                          time_range=geometry.time_axis, 
                          coordinates=geometry.rec_positions)
        grad = Function(name="grad", grid=self.initmodel.grid)
        
        loss = 0.
        if isrcs is None:
            nsrc = self.geometry.src_positions.shape[0]
            isrcs = range(nsrc)
        for isrc in tqdm(isrcs):
            # update source location in geometry
            geometry.src_positions[0, :] = self.geometry.src_positions[isrc, :]
            src.coordinates.data[0, :] = self.geometry.src_positions[isrc, :]
            lossgrad = self._loss_grad_oneshot(vp, geometry, src, solver, d_obs, d_syn, adjsrc, grad, dobs[isrc], isrc)
            if computeloss and computegrad:
                loss_isrc, grad = lossgrad
                loss += loss_isrc
            elif computeloss:
                loss += lossgrad
            else:
                grad = lossgrad

        if computegrad: 
            # Gradient postprocessing [MUST DO IT WITH FUNCTION PASSED TO CLASS]
            # Scale gradient from slowness square to chosen quantity 
            grad = - grad.data[:] / (vp.data[:] ** 3)
            
            # Extract gradient in grid
            grad = self._crop_model(grad, self.nbl)

            # Mask gradient
            if mask is not None:
                grad *= mask
        
        if computeloss and computegrad:
            return loss, grad
        elif computeloss:
            return loss
        else:
            return grad 
        
    def loss_grad(self, x, dobs, mask=None, scaling=1., computeloss=True, computegrad=True, debug=False):
        """Compute loss function and gradient to be used by solver

        This routine wraps _loss_grad providing and returning numpy arrays 
        and should be used with any solver
        
        Parameters
        ----------
        
        """
        # Convert x to velocity
        vp = x.reshape(self.initmodel.shape)
        
        # Overwrite current velocity in geometry (don't update boundary region)
        self.initmodel.update('vp', vp.reshape(self.initmodel.shape))
        
        # Evaluate objective function 
        lossgrad = self._loss_grad(self.initmodel.vp, dobs, mask=mask)
        if computeloss and computegrad:
            loss, grad = lossgrad
        elif computeloss:
            loss = lossgrad
        else:
            grad = lossgrad

        # Rescale loss and gradient
        if computeloss:
            loss /= scaling
        if computegrad:
            grad = grad.astype(np.float64) / scaling

        # Save loss history
        if computeloss:
            self.losshistory.append(loss)
        
        # Display results in debugging mode
        if debug:
            print('loss, scaling, grad.min(), grad.max()', 
                  fval, scaling, grad.min(), grad.max())
            plt.figure()
            plt.imshow(grad.T, vmin=-3, vmax=3,
                    aspect='auto', cmap='seismic')
            plt.colorbar()
        
        if computeloss and computegrad:
            return loss, grad.ravel()
        elif computeloss:
            return loss
        else:
            return grad.ravel()

    def loss(self, x, dobs, mask=None, scaling=1., debug=False):
        """Compute loss function to be used by solver

        Parameters
        ----------
        
        """
        return self.lossgrad(x, dobs, mask=mask, scaling=scaling, 
                             computeloss=True, computegrad=False, debug=debug)

    def grad(self, x, dobs, mask=None, scaling=1., debug=False):
        """Compute gradient to be used by solver

        Parameters
        ----------
        
        """
        return self.lossgrad(x, dobs, mask=mask, scaling=scaling, 
                             computeloss=False, computegrad=True, debug=debug)
