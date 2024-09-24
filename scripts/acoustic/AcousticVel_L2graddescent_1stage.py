r"""
Acoustic FWI(VP) with entire data

This example is used to showcase how to perform acoustic FWI in a distributed manner using
MPI4py.

Run as: export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mpiexec -n 8 python AcousticVel_L2graddescent_1stage.py 
"""

import os
import numpy as np
import time

from matplotlib import pyplot as plt
from mpi4py import MPI
from pylops.basicoperators import Identity
from pylops_mpi.DistributedArray import local_split, Partition

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
from devitofwi.optimization.firstorder import gradient_descent

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

configuration['log-level'] = 'ERROR'
# clear_devito_cache()

# Path to save figures
figpath = './figs/AcousticVel_L2graddescent_1stage'

if not os.path.isdir(figpath):
    os.mkdir(figpath)

# Callback to track model error
def fwi_callback(xk, vp, vp_error):
    vp_error.append(np.linalg.norm((xk - vp.reshape(-1))/vp.reshape(-1)))

    if rank == 0:
        plt.figure(figsize=(14, 5))
        plt.imshow(xk.reshape(vp.shape).T, vmin=m_vmin, vmax=m_vmax,
                cmap='jet')
        plt.colorbar()
        plt.title(f'Inverted VP (iter {len(vp_error)})')
        plt.axis('tight')
        plt.savefig(os.path.join(figpath, 'InvertedVPtmp.png'))
        plt.close('all')

if rank == 0:
    print(f'Distributed FWI with gradient descent ({size} ranks)')


##################################################################
# Parameters
##################################################################

# Model and aquisition parameters
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

if rank == 0:
    m_vmin, m_vmax = np.percentile(vp_true, [2,98]) 

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
vp_init = gaussian_filter(vp_true, sigma=[15,10])
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

# Model data
if rank == 0:
    print('Model data (and gather)...')

if rank == 0:
    print('Model data...')
dobs, dtobs = amod.mod_allshots()

##################################################################
# Gradient
##################################################################

# Define loss 
l2loss = L2(Identity(int(np.prod(dobs.shape[1:]))), dobs.reshape(ns_rank[0], -1))

ainv = AcousticWave2D(shape, origin, spacing, 
                      x_s[isin_rank:isend_rank, 0], x_s[isin_rank:isend_rank, 1], 
                      x_r[:, 0], x_r[:, 1], 
                      0., tmax,  
                      vprange=(vp_true.min(), vp_true.max()),
                      src_type="Ricker", f0=par['freq'],
                      space_order=space_order, nbl=nbl,
                      loss=l2loss,
                      base_comm=comm)

# Compute first gradient and find scaling
postproc = PostProcessVP(scaling=1, mask=msk)

if rank == 0:
    print('Compute gradient...')
    
loss, direction = ainv._loss_grad(vp_init, postprocess=postproc.apply)

scaling = direction.max()

if rank == 0:
    plt.figure(figsize=(14, 5))
    plt.imshow(direction.T / scaling, cmap='seismic', vmin=-1e-1, vmax=1e-1, 
            extent=(x[0], x[-1], z[-1], z[0]))
    plt.colorbar()
    plt.scatter(x_r[:,0], x_r[:,1], c='w')
    plt.scatter(x_s[:,0], x_s[:,1], c='r')
    plt.title('L2 Gradient')
    plt.axis('tight')
    plt.savefig(os.path.join(figpath, 'Gradient.png'))

##################################################################
# FWI
##################################################################

# Gradient descent parameters
stepsize = 1e-2
maxiter = 1000
vp_error = []

# Run FWI
convertvp = None
postproc = PostProcessVP(scaling=scaling, mask=msk)

if rank == 0:
    print('Run FWI...')
    
vp_inv = gradient_descent(ainv.loss_grad, vp_init.ravel(), stepsize=stepsize, niter=maxiter,
                          args=(convertvp, postproc.apply),
                          callback=lambda x: fwi_callback(x, vp=vp_true, vp_error=vp_error), 
                          show=True if rank ==0 else False)
if rank == 0:
    plt.figure(figsize=(14, 5))
    plt.plot(ainv.losshistory, 'k')
    plt.title('Loss history')
    plt.savefig(os.path.join(figpath, 'Loss.png'))

    plt.figure(figsize=(14, 5))
    plt.plot(vp_error, 'k')
    plt.title('Model error history')
    plt.savefig(os.path.join(figpath, 'ModelError.png'))

    vp_inv = vp_inv.reshape(shape)

    plt.figure(figsize=(14, 5))
    plt.imshow(vp_inv.T, vmin=m_vmin, vmax=m_vmax, cmap='jet', extent=(x[0], x[-1], z[-1], z[0]))
    plt.colorbar()
    plt.scatter(x_r[:,0], x_r[:,1], c='w')
    plt.scatter(x_s[:,0], x_s[:,1], c='r')
    plt.title('Inverted VP')
    plt.axis('tight')
    plt.savefig(os.path.join(figpath, 'InvertedVP.png'))
