r"""
Acoustic Gradient computation with snap-shotting

This example implements snap-shotting when computing the gradient:
in other words, instead of storing a snapshot of the source wavefield
at every time step, snapshots are stored every `factor` time steps
to reduce the memory footprint of the adjoint operator.

Memory profiling is also performed to understand the benefits of snap-shotting over
storing the wavefields along the entire time axis for gradient computation.

Memory-profiler run as: export DEVITO_LANGUAGE=openmp; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; mprof run Gradient_snap.py; mprof plot
Memray run as: export DEVITO_LANGUAGE=openmp; export OMP_NUM_THREADS=6; export MKL_NUM_THREADS=6; export NUMBA_NUM_THREADS=6; memray run Gradient_snap.py; memray summary memray-Gradient_snap*bin

"""
import time
import numpy as np
import matplotlib.pyplot as plt

from memory_profiler import memory_usage
from scipy.ndimage import gaussian_filter
from pylops.basicoperators import Identity

from devito import configuration
from tqdm.notebook import tqdm

from devitofwi.waveengine.acoustic import AcousticWave2D
from devitofwi.loss.l2 import L2
from devitofwi.postproc.acoustic import create_mask_value, PostProcessVP

configuration['log-level'] = 'ERROR'

tstart = time.time()

# Model and aquisition parameters (in km, s, and Hz units)
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

# Load the true model
vp_true = np.fromfile(velocity_file, np.float32).reshape(par['nz'], par['nx']).T
msk = create_mask_value(vp_true, 1.52) # get the mask for the water layer 

m_vmin, m_vmax = np.percentile(vp_true, [2,98]) 

plt.figure(figsize=(14, 5))
plt.imshow(vp_true.T, vmin=m_vmin, vmax=m_vmax, cmap='jet', 
           extent=(x[0], x[-1], z[-1], z[0]))
plt.colorbar()
plt.scatter(x_r[:,0], x_r[:,1], c='w')
plt.scatter(x_s[:,0], x_s[:,1], c='r')
plt.title('True VP')
plt.axis('tight');

# Initial model for FWI by smoothing the true model
vp_init = gaussian_filter(vp_true, sigma=[15,10])
vp_init = vp_init * msk  # to preserve the water layer  
vp_init[vp_init == 0] = 1.5

# Data modelling
amod = AcousticWave2D(shape, origin, spacing, 
                      x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1], 
                      0., tmax,
                      vp=vp_true, 
                      src_type="Ricker", f0=par['freq'],
                      space_order=space_order, nbl=nbl,
                      clearcache=False)

# Create model and geometry to extract useful information to define the filtering object
model, geometry = amod.model_and_geometry()

# Model data
dobs, dtobs = amod.mod_allshots()
mem_mod = memory_usage()[0]
tendmod = time.time()
print('End of modelling at time', tendmod - tstart)
print(f"Memory usage at the end of modelling: {mem_mod} MiB")

# Snapshot subsampling
factor = 8
dtsnap = geometry.dt * factor
fnyqsnap = 1. / (2 * dtsnap)

# Snapshotted gradient
l2loss = L2(Identity(int(np.prod(dobs.shape[1:]))), dobs.reshape(par['ns'], -1))

ainvsnap = AcousticWave2D(shape, origin, spacing, 
                      x_s[:, 0], x_s[:, 1], x_r[:, 0], x_r[:, 1], 
                      0., tmax,
                      vprange=(vp_true.min(), vp_true.max()),
                      src_type="Ricker", f0=par['freq'],
                      space_order=space_order, nbl=nbl,
                      loss=l2loss, clearcache=True,
                      factor=factor);

postproc = PostProcessVP(scaling=1, mask=msk)
loss_snap, direction_snap = ainvsnap._loss_grad(vp_init, postprocess=postproc.apply)
mem_gradsnap = memory_usage()[0]
tendgradsnap = time.time()
print('End of gradient with snapshots at time', tendgradsnap - tstart)
print(f"Memory usage at the end of gradient with snapshots: {mem_gradsnap} MiB")