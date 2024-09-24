r"""
Gradient computation using MPI reduction

This example is used to showcase how to compute FWI gradients in a distributed manner using
MPI4py.

Run as: export DEVITO_LANGUAGE=openmp; export DEVITO_MPI=0; export OMP_NUM_THREADS=12; export MKL_NUM_THREADS=12; export NUMBA_NUM_THREADS=12; mpiexec -n 4 python Gradient_comparison_mpi.py 
"""

import os
import numpy as np

from matplotlib import pyplot as plt
from mpi4py import MPI
from pylops.basicoperators import Identity
from pylops_mpi.DistributedArray import local_split, Partition

from scipy.ndimage import gaussian_filter
from devito import configuration
from examples.seismic import AcquisitionGeometry, Model, Receiver
from examples.seismic import plot_velocity, plot_perturbation
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import plot_shotrecord

from devitofwi.devito.utils import clear_devito_cache
from devitofwi.waveengine.acoustic import AcousticWave2D
from devitofwi.preproc.masking import TimeSpaceMasking
from devitofwi.loss.l2 import L2
from devitofwi.postproc.acoustic import PostProcessVP

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

configuration['log-level'] = 'ERROR'
# clear_devito_cache()

# Path to save figures
figpath = './figs/Gradient_comparison'

if not os.path.isdir(figpath):
    os.mkdir(figpath)

if rank == 0:
    print(f'Distributed Gradient computation ({size} ranks)')


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

# Model data (and gather to all ranks)
if rank == 0:
    print('Model data (and gather)...')

dobstot, dtobs = amod.mod_allshots_mpi()

if rank == 0:
    # Plot shot gathers
    d_vmin, d_vmax = np.percentile(np.hstack(dobstot).ravel(), [2, 98])

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(14, 9))
    for ax, ishot in zip(axs, [0, par['ns']//2, par['ns']-1]):
        ax.imshow(dobstot[ishot], aspect='auto', cmap='gray',
                  extent=(x_r[0, 0], x_r[-1, 0], tmax, 0.,),
                  vmin=-d_vmax, vmax=d_vmax)
    plt.savefig(os.path.join(figpath, 'Data.png'))

# Model data (without gathering)
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
postproc = PostProcessVP(scaling=1)

if rank == 0:
    print('Compute gradient...')
    
lossl2, directionl2 = ainv._loss_grad(vp_init, postprocess=postproc.apply)

scalingl2 = directionl2.max()

if rank == 0:
    plt.figure(figsize=(14, 5))
    plt.imshow(directionl2.T / scalingl2, cmap='seismic', vmin=-1e-1, vmax=1e-1, 
            extent=(x[0], x[-1], z[-1], z[0]))
    plt.colorbar()
    plt.scatter(x_r[:,0], x_r[:,1], c='w')
    plt.scatter(x_s[:,0], x_s[:,1], c='r')
    plt.title('L2 Gradient')
    plt.axis('tight')
    plt.savefig(os.path.join(figpath, 'Gradient.png'))

# Check gradient against single rank implementation
if rank == 0:
    print('Compute gradient (single ranks)...')
    
    # Define loss 
    l2loss = L2(Identity(int(np.prod(dobstot.shape[1:]))), dobstot.reshape(par['ns'], -1))
    
    wav = geometry.src.wavelet
    ainv = AcousticWave2D(shape, origin, spacing, 
                          x_s[:, 0], x_s[:, 1], 
                          x_r[:, 0], x_r[:, 1], 
                          0., tmax,  
                          vprange=(vp_true.min(), vp_true.max()),
                          src_type="Ricker", f0=par['freq'],
                          space_order=space_order, nbl=nbl,
                          loss=l2loss,
                          base_comm=None)

    # Compute first gradient and find scaling
    postproc = PostProcessVP(scaling=1)
    
    lossl2_single, directionl2_single = ainv._loss_grad(vp_init, postprocess=postproc.apply)
    
    scalingl2_single = directionl2_single.max()

    print(f'LossL2: multi {lossl2}, single {lossl2_single}')

    plt.figure(figsize=(14, 5))
    plt.imshow(directionl2_single.T / scalingl2_single, cmap='seismic', vmin=-1e-1, vmax=1e-1,
               extent=(x[0], x[-1], z[-1], z[0]))
    plt.colorbar()
    plt.scatter(x_r[:,0], x_r[:,1], c='w')
    plt.scatter(x_s[:,0], x_s[:,1], c='r')
    plt.title('L2 Gradient')
    plt.axis('tight')
    plt.savefig(os.path.join(figpath, 'GradientSingle.png'))

    plt.figure(figsize=(14, 5))
    plt.imshow(directionl2.T / scalingl2 - directionl2_single.T / scalingl2_single, cmap='seismic',
               extent=(x[0], x[-1], z[-1], z[0]))
    plt.colorbar()
    plt.scatter(x_r[:,0], x_r[:,1], c='w')
    plt.scatter(x_s[:,0], x_s[:,1], c='r')
    plt.title('Gradient Multi-Single')
    plt.axis('tight')
    plt.savefig(os.path.join(figpath, 'GradientDiff.png'))
