r"""
Acoustic FWI(VP) with entire data and TV regularization

This example is used to showcase how to perform acoustic FWI with TV regularization 
in a distributed manner coupling MPI4py with PyProximal.

Run as: export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 8 python AcousticVel_L2PDTV_1stage.py 
"""

import os
import numpy as np

from matplotlib import pyplot as plt
from mpi4py import MPI
from pylops.basicoperators import Diagonal, Gradient
from pylops_mpi.DistributedArray import local_split, Partition
from pyproximal.proximal import *
from pyproximal.optimization.primal import *
from pyproximal.optimization.primaldual import *

from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from devito import configuration
from examples.seismic import AcquisitionGeometry, Model, Receiver
from examples.seismic import plot_velocity, plot_perturbation
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import plot_shotrecord

from devitofwi.devito.utils import clear_devito_cache
from devitofwi.waveengine.acoustic import AcousticWave2D
from devitofwi.preproc.masking import TimeSpaceMasking
from devitofwi.loss.l2 import L2
from devitofwi.postproc.acoustic import create_mask_value, PostProcessVP

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

configuration['log-level'] = 'ERROR'
# clear_devito_cache()

# Path to save figures
figpath = './figs/AcousticVel_L2PDTV_1stage'

if rank == 0:
    if not os.path.isdir(figpath):
        os.mkdir(figpath)

# Callback to track model error
def fwi_callback(xk, vp, vp_error, vp_tmp, m_vmin, m_vmax, rank):
    vp_tmp[0] = xk
    vp_error.append(np.linalg.norm((xk - vp.reshape(-1)) / vp.reshape(-1)))

    if rank == 0:
        plt.figure(figsize=(14, 5))
        plt.imshow(xk.reshape(vp.shape).T, vmin=m_vmin, vmax=m_vmax, 
                cmap='jet')
        plt.colorbar()
        plt.title(f'Inverted VP (iter {len(vp_error)})')
        plt.axis('tight')
        plt.savefig(os.path.join(figpath, 'InvertedVPtmp.png'))
        plt.close('all')

def pd_callback(xk, vp, vp_error, m_vmin, m_vmax, rank):
    vp_error.append(np.linalg.norm((xk - vp.reshape(-1))/vp.reshape(-1)))

    if rank == 0:
        plt.figure(figsize=(14, 5))
        plt.imshow(xk.reshape(vp.shape).T, vmin=m_vmin, vmax=m_vmax, 
                   cmap='jet')
        plt.colorbar()
        plt.title(f'Inverted VP (PD iter {len(vp_error)})')
        plt.axis('tight')
        plt.savefig(os.path.join(figpath, 'InvertedVPpdtmp.png'))
        plt.close('all')

# Nonlinear pyprox wrapper
class AcousticWave2Dwrapper(Nonlinear):
    def setup(self, awop, convertvp, postproc, 
              ftol=1e-10, maxfun=5000, maxls=5, disp=False, 
              vp_true=None, vp_error=None, vp_lims=None, rank=None):
        self.awop = awop
        self.convertvp = convertvp
        self.postproc = postproc
        self.ftol, self.maxfun, self.maxls = ftol, maxfun, maxls
        self.disp = disp
        self.vp_true, self.vp_error = vp_true, vp_error
        self.vp_tmp = [0, ]
        self.vp_lims = vp_lims
        self.rank = rank
        self.flast = 0.

    def fungrad(self, x):
        self.flast, self.glast = self.awop.loss_grad(x, self.convertvp, self.postproc.apply)
        return self.flast, self.glast
    
    def fun(self, x):
        return self.flast
    
    def optimize(self):
        callback = None
        if self.vp_true is not None:
            callback=lambda x: fwi_callback(x, vp=self.vp_true, vp_error=self.vp_error, vp_tmp=self.vp_tmp, 
                                            m_vmin=self.vp_lims[0], m_vmax=self.vp_lims[1], rank=self.rank)
        sol = minimize(lambda x: self._fungradprox(x, self.tau),
                       x0=self.x0,
                       method='L-BFGS-B', jac=True, 
                       callback=callback,
                       options={'ftol':self.ftol, 'maxiter':self.niter, 
                                'maxfun':self.maxfun, 'maxls':self.maxls,
                                'disp':self.disp})
        if self.disp:
            print(sol)
        sol = sol.x
        return sol

if rank == 0:
    print(f'Distributed FWI ({size} ranks)')


##################################################################
# Parameters
##################################################################

# Model and acquisition parameters (in km, s, and Hz units)
par = {'nx':601,   'dx':0.015,    'ox':0,
       'nz':221,   'dz':0.015,    'oz':0,
       'ns':20,    'ds':0.300,    'os':1.,  'sz':0,
       'nr':300,   'dr':0.030,    'or':0,   'rz':0,
       'nt':3000,  'dt':0.002, 'ot':0,
       'freq':15,
      }

# Modelling parameters
shape = (par['nx'], par['nz'])
spacing = (par['dx'], par['dz'])
origin = (par['ox'], par['oz'])
space_order = 4
nbl = 20

# Velocity model
path = '../../data/'
velocity_file = path + 'Marm.bin'

# Time-space mask parameters
vwater = 1.5
toff = 0.45

##################################################################
# Acquisition set-up
##################################################################

# Sampling frequency
fs = 1 / par['dt'] 

# Axes
x = np.arange(par['nx']) * par['dx'] + par['ox']
z = np.arange(par['nz']) * par['dz'] + par['oz']
t = np.arange(par['nt']) * par['dt'] + par['ot']
tmax = t[-1]

# Sources
x_s = np.zeros((par['ns'], 2))
x_s[:, 0] = np.arange(par['ns']) * par['ds'] + par['os']
x_s[:, 1] = par['sz']

# Receivers
x_r = np.zeros((par['nr'], 2))
x_r[:, 0] = np.arange(par['nr']) * par['dr'] + par['or']
x_r[:, 1] = par['rz']

##################################################################
# Velocity model
##################################################################

# Load the true model
vp_true = np.fromfile(velocity_file, np.float32).reshape(par['nz'], par['nx']).T
msk = create_mask_value(vp_true, 1.52) # get the mask for the water layer
m_vmin, m_vmax = np.percentile(vp_true, [2, 98])

if rank == 0:
    plt.figure(figsize=(14, 5))
    plt.imshow(vp_true.T, vmin=m_vmin, vmax=m_vmax, cmap='jet', 
            extent=(x[0], x[-1], z[-1], z[0]))
    plt.colorbar()
    plt.scatter(x_r[:,0], x_r[:,1], c='w')
    plt.scatter(x_s[:,0], x_s[:,1], c='r')
    plt.title('True VP')
    plt.axis('tight')
    plt.savefig(os.path.join(figpath, 'TrueVel.png'))

# Initial model for FWI by smoothing the true model
vp_init = gaussian_filter(vp_true, sigma=[15, 10])
vp_init = vp_init * msk  # to preserve the water layer  
vp_init[vp_init == 0] = 1.5

if rank == 0:
    plt.figure(figsize=(14, 5))
    plt.imshow(vp_init.T, vmin=m_vmin, vmax=m_vmax, cmap='jet', 
    extent=(x[0], x[-1], z[-1], z[0]))
    plt.colorbar()
    plt.scatter(x_r[:,0], x_r[:,1], c='w')
    plt.scatter(x_s[:,0], x_s[:,1], c='r')
    plt.title('Initial VP')
    plt.axis('tight')
    plt.savefig(os.path.join(figpath, 'InitialVel.png'))

##################################################################
# Data
##################################################################

# Choose how to split sources to ranks
ns_rank = local_split((par['ns'], ), MPI.COMM_WORLD, Partition.SCATTER, 0)
ns_ranks = np.concatenate(MPI.COMM_WORLD.allgather(ns_rank))
isin_rank = np.insert(np.cumsum(ns_ranks)[:-1] , 0, 0)[rank]
isend_rank = np.cumsum(ns_ranks)[rank]
print(f'Rank: {rank}, ns: {ns_rank}, isin: {isin_rank}, isend: {isend_rank}')

# Define modelling engine
amod = AcousticWave2D(shape, origin, spacing, 
                      x_s[isin_rank:isend_rank, 0], x_s[isin_rank:isend_rank, 1], 
                      x_r[:, 0], x_r[:, 1], 
                      0., tmax,  
                      vp=vp_true,
                      src_type="Ricker", f0=par['freq'],
                      space_order=space_order, nbl=nbl,
                      base_comm=comm)

# Create model and geometry to extract useful information to define the filtering object
model, geometry = amod.model_and_geometry()

# Model data
if rank == 0:
    print('Model data (and gather)...')

if rank == 0:
    print('Model data...')
dobs, dtobs = amod.mod_allshots()

# Compute gain and apply to observed data
tobs = geometry.time_axis.time_values
gain = np.repeat(tobs[:, None], par['nr'], axis=-1)
dobsgain = dobs * gain

if rank == 0:
    # Plot shot gathers
    d_vmin, d_vmax = np.percentile(np.hstack(dobs).ravel(), [2, 98])
    dg_vmin, dg_vmax = np.percentile(np.hstack(dobsgain).ravel(), [2, 98])

    fig, axs = plt.subplots(1, 3, figsize=(14, 9))
    for ax, ishot in zip(axs, [0, ns_ranks[rank]//2, ns_ranks[rank]-1]):
        ax.imshow(dobs[ishot], aspect='auto', cmap='gray',
                  vmin=-d_vmax, vmax=d_vmax)
    plt.savefig(os.path.join(figpath, 'Data.png'))

    fig, axs = plt.subplots(1, 3, figsize=(14, 9))
    for ax, ishot in zip(axs, [0, ns_ranks[rank]//2, ns_ranks[rank]-1]):
        ax.imshow(dobsgain[ishot], aspect='auto', cmap='gray',
                  vmin=-dg_vmax, vmax=dg_vmax)
    plt.savefig(os.path.join(figpath, 'Datagain.png'))

##################################################################
# FWI
##################################################################

# Parameters
niter_inner = 10  # number of inner iterations of L-BFGS for f function
niter = 100  # number of outer iterations for PD
sigma = 0.01  # scaling factor of TV
# scaling factor of f (must be comparable to that of TV to enable TV to act 
# on the solution... this is needed because the original scaling of the f function
# is very large and would not allow TV to contribute)
scaling = 1e3

# Define loss 
Gainop = Diagonal(gain.ravel())
l2loss = L2(Gainop, dobsgain.reshape(ns_rank[0], -1))

ainv = AcousticWave2D(shape, origin, spacing, 
                      x_s[isin_rank:isend_rank, 0], x_s[isin_rank:isend_rank, 1], 
                      x_r[:, 0], x_r[:, 1], 
                      0., tmax,  
                      vprange=(vp_true.min(), vp_true.max()),
                      src_type="Ricker", f0=par['freq'],
                      space_order=space_order, nbl=nbl,
                      loss=l2loss,
                      base_comm=comm)

# Nonlinear objective
vp_error = []
postproc = PostProcessVP(scaling=scaling, mask=msk)
nl = AcousticWave2Dwrapper(vp_init.ravel(), niter=niter_inner, warm=True)
nl.setup(ainv, None, postproc, ftol=1e-10, maxfun=niter_inner, maxls=2,
         disp=True if rank == 0 else False, 
         vp_true=vp_true, vp_error=vp_error,
         vp_lims=(m_vmin, m_vmax), rank=rank)

# PD with TV regularization
l21 = L21(ndim=2, sigma=sigma)
Dop = Gradient(dims=shape, edge=True, dtype=np.float64, kind='forward')

L = 8.  # np.real((Dop.H*Dop).eigs(neigs=1, which='LM')[0])
tau = 1.
mu = 0.99 / (tau * L)

vp_inverror = []
vp_inv = PrimalDual(nl, l21, Dop, x0=vp_init.ravel(), tau=tau, mu=mu, 
                    theta=1., niter=niter, show=True if rank == 0 else False,
                    callback=lambda x: pd_callback(x, vp_true, vp_inverror, m_vmin, m_vmax, rank))
vp_inv = vp_inv.reshape(shape)

if rank == 0:
    plt.figure(figsize=(14, 5))
    plt.plot(ainv.losshistory, 'k')
    plt.title('Loss history')
    plt.savefig(os.path.join(figpath, 'Loss.png'))

    plt.figure(figsize=(14, 5))
    plt.plot(vp_error, 'k')
    plt.title('Model error history')
    plt.savefig(os.path.join(figpath, 'ModelErrorAlliters.png'))

    plt.figure(figsize=(14, 5))
    plt.plot(vp_inverror, 'k')
    plt.title('Model error history')
    plt.savefig(os.path.join(figpath, 'ModelError.png'))

    plt.figure(figsize=(14, 5))
    plt.imshow(vp_inv.T, vmin=m_vmin, vmax=m_vmax, cmap='jet', extent=(x[0], x[-1], z[-1], z[0]))
    plt.colorbar()
    plt.scatter(x_r[:,0], x_r[:,1], c='w')
    plt.scatter(x_s[:,0], x_s[:,1], c='r')
    plt.title('Inverted VP')
    plt.axis('tight')
    plt.savefig(os.path.join(figpath, 'InvertedVP.png'))
