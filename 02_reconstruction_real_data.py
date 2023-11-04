"""script for Na AGR reconstruction of pre-processed raw data

The input data of a subject should be organized as follows:

subject_directory
|-- anatomical_prior_image.nii
|
|-- kspace_trajectory.h5
|
|-- raw_echo1
|   |-- converted_data.h5
|
`-- raw_echo2
    |-- converted_data.h5
"""

from __future__ import annotations

import h5py
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import scipy.ndimage as ndi
import sigpy
import pymirc.viewer as pv
import nibabel as nib
import SimpleITK as sitk

from copy import deepcopy
from pathlib import Path

from utils import align_images
from utils_sigpy import NUFFTT2starDualEchoModel, projected_gradient_operator

import argparse

parser = argparse.ArgumentParser(
    description='AGR sodium recons of real data',
)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--beta_non_anatomical', type=float, default=1e-1)
parser.add_argument('--beta_anatomical', type=float, default=3e-3)
parser.add_argument('--beta_r', type=float, default=3e-1)
parser.add_argument('--num_iter_r', type=int, default=100)
parser.add_argument('--max_num_iter', type=int, default=2000)
parser.add_argument('--odir', type=str, default=None)
parser.add_argument('--matrix_size', type=int, default=128)
parser.add_argument('--eta', type=float, default=0.005)
parser.add_argument('--alpha', type=float, default=1.0)
args = parser.parse_args()

#--------------------------------------------------------------------
# input parameters
data_dir = Path(args.data_dir)

beta_anatomical = args.beta_anatomical
beta_non_anatomical = args.beta_non_anatomical
max_num_iter = args.max_num_iter
beta_r = args.beta_r
num_iter_r = args.num_iter_r
matrix_size = args.matrix_size
eta = args.eta

# step size for the gradient descent in ratio update
alpha = args.alpha
#--------------------------------------------------------------------
# fixed parameters
t1_file = data_dir / 'anatomical_prior_image.nii'

# shape of the images to be reconstructed
ishape = (matrix_size, matrix_size, matrix_size)

# regularization parameters for non-anatomy-guided recons
regularization_norm_non_anatomical = 'L2'

# regularization parameters for anatomy-guided recons
regularization_norm_anatomical = 'L1'

# sigma step size of PDHG
sigma = 0.1

# time bin width for T2* decay modeling
time_bin_width_ms: float = 0.25

# echo times in ms
echo_time_1_ms = 0.455
echo_time_2_ms = 5.

# sampling time in us and ms
dt_us = 10.
acq_sampling_time_ms = dt_us * 1e-3

# field of view in cm
field_of_view_cm = 22.

# show the readout data (to spot bad readouts)
show_readouts = False

# create the output directory
if args.odir is None:
    odir = Path(data_dir) / f'zz_recons_{matrix_size}'
else:
    odir = Path(args.odir)
odir.mkdir(exist_ok=True, parents=True)

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# read the k-space trajectory
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# load the (pre-processed) kspace trajectory in 1/cm
with h5py.File(data_dir / 'kspace_trajectory.h5', 'r') as f:
    k_1_cm = f['k'][...]

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# read the acquired data
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# load the multi-channel non-uniform k-space data
# the array will have shape (num_channels, num_points, num_readouts)
# we only read the data from the 1st coil
with h5py.File(data_dir / 'raw_echo1' / 'converted_data.h5', 'r') as f1:
    data_echo_1 = f1['data'][0,:,:]

with h5py.File(data_dir / 'raw_echo2' / 'converted_data.h5', 'r') as f2:
    data_echo_2 = f2['data'][0,:,:]

if not np.iscomplexobj(data_echo_1):
    data_echo_1 = data_echo_1['real'] + 1j * data_echo_1['imag']

if not np.iscomplexobj(data_echo_2):
    data_echo_2 = data_echo_2['real'] + 1j * data_echo_2['imag']

# send data to the GPU
data_echo_1 = cp.asarray(data_echo_1)
data_echo_2 = cp.asarray(data_echo_2)


# calculate the signal max across all readouts to detect potential readout problems
max_echo_1 = cp.asnumpy(cp.abs(data_echo_1).max(0))
max_echo_2 = cp.asnumpy(cp.abs(data_echo_2).max(0))

# calculate the most common maximum signal to find outliers
h1 = np.histogram(ndi.gaussian_filter1d(max_echo_1, 5), bins=100)
most_common_max_1 = h1[1][np.argmax(h1[0])]
h2 = np.histogram(ndi.gaussian_filter1d(max_echo_2, 5), bins=100)
most_common_max_2 = h2[1][np.argmax(h2[0])]

th1 = most_common_max_1 - 3 * max_echo_1[max_echo_1 >= 0.95 *
                                         most_common_max_1].std()
th2 = most_common_max_2 - 3 * max_echo_2[max_echo_2 >= 0.95 *
                                         most_common_max_2].std()

i_bad_1 = np.where(max_echo_1 < th1)[0]
i_bad_2 = np.where(max_echo_2 < th2)[0]

print(f'num bad readouts 1st echo {i_bad_1.size}')
print(f'num bad readouts 2nd echo {i_bad_2.size}')

# setup the data weights (1 for good reaouts, 0 for bad readouts)
data_weights_1 = cp.ones(data_echo_1.shape, dtype=np.uint8)
data_weights_2 = cp.ones(data_echo_2.shape, dtype=np.uint8)

data_weights_1[:, i_bad_1] = 0
data_weights_2[:, i_bad_2] = 0
# also ignore the first readout point
data_weights_1[0, :] = 0
data_weights_2[0, :] = 0

# set data bins of bad readouts to 0
data_echo_1 *= data_weights_1
data_echo_2 *= data_weights_2

# print info related to SNR
for i in np.linspace(0, data_echo_1.shape[1], 10, endpoint=False).astype(int):
    print(
        f'{i:04} {float(cp.abs(data_echo_1[:, i]).max() / cp.abs(data_echo_1[-100:, i]).std()):.2f}'
    )

# scale the data such that we get CSF approx 3 with normalized nufft operator
data_scale = 2. / cp.abs(data_echo_1)[1, :].mean()

data_echo_1 *= data_scale
data_echo_2 *= data_scale

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# gridded recon of both echos
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# ignore last data points in kspace trajectory
# (contains more points compared to data points)
k_1_cm = k_1_cm[:data_echo_1.shape[0], ...]

# flatten the data for the nufft operators
data_echo_1 = data_echo_1.ravel()
data_echo_2 = data_echo_2.ravel()
data_weights_1 = data_weights_1.ravel()
data_weights_2 = data_weights_2.ravel()

# setup the density compensation weights
abs_k = np.linalg.norm(k_1_cm, axis=-1)
dk = abs_k[1:,0] - abs_k[:-1,0]
ilin = np.where(dk > 0.99*dk.max())[0].max()
abs_k_twist = abs_k[ilin, 0]

dcf = cp.asarray(np.clip(abs_k**2, None, abs_k_twist**2)).ravel()

ifft1 = sigpy.nufft_adjoint(data_echo_1 * dcf,
                            cp.array(k_1_cm).reshape(-1, 3) * field_of_view_cm,
                            (64,64,64))

ifft2 = sigpy.nufft_adjoint(data_echo_2 * dcf,
                            cp.array(k_1_cm).reshape(-1, 3) * field_of_view_cm,
                            (64,64,64))

# post-smoothing of gridded recons
ifft1_sm = ndimage.gaussian_filter(ifft1, 1.)
ifft2_sm = ndimage.gaussian_filter(ifft2, 1.)

# interpolate ifft to recon grid
ifft1 = ndimage.zoom(ifft1, ishape[0] / 64, order=1, prefilter=False)
ifft2 = ndimage.zoom(ifft2, ishape[0] / 64, order=1, prefilter=False)

ifft1_sm = ndimage.zoom(ifft1_sm, ishape[0] / 64, order=1, prefilter=False)
ifft2_sm = ndimage.zoom(ifft2_sm, ishape[0] / 64, order=1, prefilter=False)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# setup (unscaled) acquisition model
acq_model = NUFFTT2starDualEchoModel(ishape,
                                     k_1_cm,
                                     field_of_view_cm=field_of_view_cm,
                                     acq_sampling_time_ms=acq_sampling_time_ms,
                                     time_bin_width_ms=time_bin_width_ms,
                                     echo_time_1_ms=echo_time_1_ms,
                                     echo_time_2_ms=echo_time_2_ms)

# estimate phases and include phases into the forward model
phase_fac_1 = cp.exp(1j * cp.angle(ifft1_sm))
phase_fac_2 = cp.exp(1j * cp.angle(ifft2_sm))
acq_model.phase_factor_1 = phase_fac_1
acq_model.phase_factor_2 = phase_fac_2

# set the data weights to exclude bad readouts
acq_model.data_weights_1 = data_weights_1
acq_model.data_weights_2 = data_weights_2

nufft_norm_file = odir / 'nufft_norm.txt'

if not nufft_norm_file.exists():
    # get a test single echo nufft operator without T2* decay modeling
    # and estimate its norm
    test_nufft, test_nufft2 = acq_model.get_operators_wo_decay_model()
    max_eig_nufft_single = sigpy.app.MaxEig(test_nufft.H * test_nufft,
                                            dtype=cp.complex128,
                                            device=data_echo_1.device,
                                            max_iter=30).run()

    with open(nufft_norm_file, 'w') as f:
        f.write(f'{max_eig_nufft_single}\n')

    del test_nufft
    del test_nufft2
else:
    with open(nufft_norm_file, 'r') as f:
        max_eig_nufft_single = float(f.readline())

# scale the acquisition model such that norm of the single echo operator is 1
acq_model.scale = 1 / np.sqrt(max_eig_nufft_single)

# setup scaled single echo nufft operator
nufft_echo1_no_decay, nufft_echo2_no_decay = acq_model.get_operators_wo_decay_model(
)

# set up the operator for regularization
G = (1 / np.sqrt(12)) * sigpy.linop.FiniteDifference(ishape, axes=None)
#------------------------------------------------------
# reconstruct the first echo without T2* decay modeling

A = sigpy.linop.Vstack([nufft_echo1_no_decay, G])
A2 = sigpy.linop.Vstack([nufft_echo2_no_decay, G])

if regularization_norm_non_anatomical == 'L2':
    proxg = sigpy.prox.L2Reg(G.oshape, lamda=beta_non_anatomical)
elif regularization_norm_non_anatomical == 'L1':
    proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta_non_anatomical)
else:
    raise ValueError('unknown regularization norm')

# estimate norm of the nufft operator if not given
proxfc1 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.Conj(proxg)
])
u1 = cp.zeros(A.oshape, dtype=data_echo_1.dtype)

outfile1na = odir / f'recon_echo_1_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}_{max_num_iter}.npz'

if not outfile1na.exists():
    max_eig_wo_decay = sigpy.app.MaxEig(A.H * A,
                                        dtype=cp.complex128,
                                        device=data_echo_1.device,
                                        max_iter=30).run()
    alg1 = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfc1,
                                              proxg=sigpy.prox.NoOp(A.ishape),
                                              A=A,
                                              AH=A.H,
                                              x=deepcopy(ifft1_sm),
                                              u=u1,
                                              tau=1 /
                                              (max_eig_wo_decay * sigma),
                                              sigma=sigma,
                                              max_iter=max_num_iter)

    print('recon echo 1 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg1.update()
    print('')

    cp.savez(outfile1na, x=alg1.x, u=u1, max_eig=max_eig_wo_decay)
    recon_echo_1_wo_decay_model = alg1.x
else:
    d1 = cp.load(outfile1na)
    recon_echo_1_wo_decay_model = d1['x']
    u1 = d1['u']
    max_eig_wo_decay = float(d1['max_eig'])

#-----------------------------------------------------
proxfc2 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(proxg)
])
u2 = cp.zeros(A2.oshape, dtype=data_echo_2.dtype)

outfile2 = odir / f'recon_echo_2_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}_{max_num_iter}.npz'

if not outfile2.exists():
    alg2 = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfc2,
                                              proxg=sigpy.prox.NoOp(A2.ishape),
                                              A=A2,
                                              AH=A2.H,
                                              x=deepcopy(ifft2_sm),
                                              u=u2,
                                              tau=1. /
                                              (max_eig_wo_decay * sigma),
                                              sigma=sigma,
                                              max_iter=max_num_iter)

    print('recon echo 2 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg2.update()
    print('')

    cp.savez(outfile2, x=alg2.x, u=u2)
    recon_echo_2_wo_decay_model = alg2.x
else:
    d2 = cp.load(outfile2)
    recon_echo_2_wo_decay_model = d2['x']
    u2 = d2['u']

del A
del A2

#---------------------------------------------------------------------
#---------------------------------------------------------------------
# load and align the 1H MR image
#---------------------------------------------------------------------
#---------------------------------------------------------------------

t1_nii = nib.load(t1_file)
t1_nii = nib.as_closest_canonical(t1_nii)
t1 = t1_nii.get_fdata()

t1_affine = t1_nii.affine
t1_voxsize = t1_nii.header['pixdim'][1:4]
t1_origin = t1_affine[:-1, -1]

t1_reoriented = t1

na_voxsize = 10 * field_of_view_cm / np.array(ishape)
na_origin = t1_origin.copy()
# save the Na origin such that we can later transform the
# Na recons back to other grids
np.savetxt(odir / 'na_origin.txt', na_origin)

t1_aligned_file = odir / 't1_aligned.npy'

if not t1_aligned_file.exists():
    print('aligning the 1H MR image')
    t1_aligned, final_transform = align_images(
        np.abs(cp.asnumpy(recon_echo_1_wo_decay_model)), t1_reoriented,
        na_voxsize, na_origin, t1_voxsize, t1_origin)

    # send aligned image to GPU
    t1_aligned = cp.asarray(t1_aligned)
    cp.save(t1_aligned_file, t1_aligned)
    sitk.WriteTransform(final_transform, str(odir / 't1_transform.tfm'))
else:
    t1_aligned = cp.load(t1_aligned_file)

# normalize the intensity of the aligned t1 image (not needed, just for convenience)
t1_aligned /= cp.percentile(t1_aligned, 99.9)


# projected gradient operator that we need for DTV
PG = projected_gradient_operator(ishape, t1_aligned, eta=eta)

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

A = sigpy.linop.Vstack([nufft_echo1_no_decay, PG])
A2 = sigpy.linop.Vstack([nufft_echo2_no_decay, PG])

if regularization_norm_anatomical == 'L2':
    proxg = sigpy.prox.L2Reg(G.oshape, lamda=beta_anatomical)
elif regularization_norm_anatomical == 'L1':
    proxg = sigpy.prox.L1Reg(G.oshape, lamda=beta_anatomical)
else:
    raise ValueError('unknown regularization norm')

proxfc1 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.Conj(proxg)
])

outfile1 = odir / f'agr_echo_1_no_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{max_num_iter}.npz'

if not outfile1.exists():
    alg1 = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc1,
        proxg=sigpy.prox.NoOp(A.ishape),
        A=A,
        AH=A.H,
        x=deepcopy(recon_echo_1_wo_decay_model),
        u=u1,
        tau=1. / (max_eig_wo_decay * sigma),
        sigma=sigma,
        max_iter=max_num_iter)

    print('AGR echo 1 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg1.update()
    print('')

    cp.savez(outfile1, x=alg1.x, u=alg1.u)
    agr_echo_1_wo_decay_model = alg1.x
else:
    d1 = cp.load(outfile1)
    agr_echo_1_wo_decay_model = d1['x']
    u1 = d1['u']

#---------------------------------------------------------------------
#----------------------------------------------------------------------
# AGR of 2nd echo without decay modeling

proxfc2 = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(proxg)
])

outfile2 = odir / f'agr_echo_2_no_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{max_num_iter}.npz'

if not outfile2.exists():
    alg2 = sigpy.alg.PrimalDualHybridGradient(
        proxfc=proxfc2,
        proxg=sigpy.prox.NoOp(A2.ishape),
        A=A2,
        AH=A2.H,
        x=deepcopy(recon_echo_2_wo_decay_model),
        u=u2,
        tau=1. / (max_eig_wo_decay * sigma),
        sigma=sigma,
        max_iter=max_num_iter)

    print('AGR echo 2 - no T2* modeling')
    for i in range(max_num_iter):
        print(f'{(i+1):04} / {max_num_iter:04}', end='\r')
        alg2.update()
    print('')

    cp.savez(outfile2, x=alg2.x, u=alg2.u)
    agr_echo_2_wo_decay_model = alg2.x
else:
    d2 = cp.load(outfile2)
    agr_echo_2_wo_decay_model = d2['x']
    u2 = d2['u']

del A
del A2
del nufft_echo1_no_decay
del nufft_echo2_no_decay

#-------------------------------------------------------------------------
# calculate the ratio between the two recons without T2* decay modeling
# to estimate a monoexponential T2*

est_ratio = cp.clip(
    cp.abs(agr_echo_2_wo_decay_model) / cp.abs(agr_echo_1_wo_decay_model), 0,
    1)
# set ratio to one in voxels where there is low signal in the first echo
mask = 1 - (cp.abs(agr_echo_1_wo_decay_model)
            < 0.05 * cp.abs(agr_echo_1_wo_decay_model).max())

label, num_label = ndimage.label(mask == 1)
size = np.bincount(label.ravel())
biggest_label = size[1:].argmax() + 1
clump_mask = (label == biggest_label)

est_ratio[clump_mask == 0] = 1

init_est_ratio = est_ratio.copy()
cp.save(odir / f'init_est_ratio_{beta_anatomical:.1E}.npy', init_est_ratio)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# recons with "estimated" decay model and anatomical prior using data from both echos
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

# step size for gradient descent on ratio image
acq_model.dual_echo_data = cp.concatenate([data_echo_1, data_echo_2])

agr_both_echos_w_decay_model = deepcopy(agr_echo_1_wo_decay_model)

proxfcb = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(proxg)
])

num_outer = max_num_iter // num_iter_r

for i_outer in range(num_outer):
    print(f'outer iteration {i_outer+1} / {num_outer}')
    # regenerate recon operators with updated estimated ratio for T2* decay modeling
    recon_operator_1, recon_operator_2 = acq_model.get_operators_w_decay_model(
        est_ratio)
    A = sigpy.linop.Vstack([recon_operator_1, recon_operator_2, PG])

    if i_outer == 0:
        ub = cp.concatenate([
            u1[:data_echo_1.size], u1[:data_echo_1.size], u1[data_echo_1.size:]
        ])

    outfileb = odir / f'agr_both_echo_w_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_r:.1E}_{max_num_iter}_{i_outer}.npz'

    if not outfileb.exists():
        max_eig_w_decay = sigpy.app.MaxEig(A.H * A,
                                           dtype=cp.complex128,
                                           device=data_echo_1.device,
                                           max_iter=30).run()

        algb = sigpy.alg.PrimalDualHybridGradient(
            proxfc=proxfcb,
            proxg=sigpy.prox.NoOp(A.ishape),
            A=A,
            AH=A.H,
            x=agr_both_echos_w_decay_model,
            u=ub,
            tau=1. / (max_eig_w_decay * sigma),
            sigma=sigma,
            max_iter=num_iter_r)

        print('AGR both echos - "estimated" T2* modeling')
        for i in range(num_iter_r):
            print(f'{(i+1):04} / {num_iter_r:04}', end='\r')
            algb.update()
        print('')

        cp.savez(outfileb, x=algb.x, u=ub, max_eig=max_eig_w_decay)
        agr_both_echos_w_decay_model = algb.x
    else:
        db = cp.load(outfileb)
        agr_both_echos_w_decay_model = db['x']
        ub = db['u']
        max_eig_w_decay = float(db['max_eig'])

    del A
    del recon_operator_1
    del recon_operator_2

    #---------------------------------------------------------------------------------------
    # optimize the ratio image using gradient descent on data fidelity + prior
    #---------------------------------------------------------------------------------------

    acq_model.x = agr_both_echos_w_decay_model
    outfile_r = odir / f'est_ratio_{beta_anatomical:.1E}_{beta_r:.1E}_{num_iter_r}_{i_outer}.npy'

    if not outfile_r.exists():
        print('updating ratio image')
        # projected gradient descent on ratio image
        for i in range(num_iter_r):
            print(f'{(i+1):04} / {num_iter_r:04}', end='\r')

            # gradient based on data fidelity
            grad_df = acq_model.data_fidelity_gradient_r(est_ratio)
            # gradient of beta_r * ||PG(r)||_2^2
            grad_prior = beta_r * PG.H(PG(est_ratio))
            est_ratio = cp.clip(est_ratio - alpha * (grad_df + grad_prior),
                                1e-2, 1)

        cp.save(outfile_r, est_ratio)
    else:
        est_ratio = cp.load(outfile_r)

#-----------------------------------------------------------------------------
# visualize results

a = cp.asnumpy(cp.abs(agr_echo_1_wo_decay_model))
b = cp.asnumpy(cp.abs(agr_both_echos_w_decay_model))
c = cp.asnumpy(est_ratio)

d = cp.asnumpy(cp.abs(ifft1_sm))
e = cp.asnumpy(cp.abs(recon_echo_1_wo_decay_model))
f = cp.asnumpy(t1_aligned)

# re-orient to LPS
a = np.flip(a, (0, 1, 2))
b = np.flip(b, (0, 1, 2))
c = np.flip(c, (0, 1, 2))
d = np.flip(d, (0, 1, 2))
e = np.flip(e, (0, 1, 2))
f = np.flip(f, (0, 1, 2))

ims1 = 2 * [dict(vmin=0, vmax=2, cmap='Greys_r')] + [
    dict(cmap='Greys_r', vmin=0, vmax=1)
]
vi1 = pv.ThreeAxisViewer([
    a,
    b,
    c,
], imshow_kwargs=ims1, sl_z=76)
vi1.fig.savefig(outfileb.with_suffix('.png'), dpi=300)

ims2 = 2 * [dict(vmin=0, vmax=2, cmap='Greys_r')] + [dict(cmap='Greys_r')]
vi2 = pv.ThreeAxisViewer([
    d,
    e,
    f,
], imshow_kwargs=ims2, sl_z=76)
vi2.fig.savefig(outfile1na.with_suffix('.png'), dpi=300)

#---------------------------------------------------------------------------------------

slz = 81
sly = 55
slx = 59

vmax = 2.

fig, ax = plt.subplots(2, 5, figsize=(5 * 2, 2 * 2))
ax[0, 0].imshow(f[:, :, slz].T, cmap='Greys_r', vmin=0, vmax=1.5)
ax[1, 0].imshow(f[slx, :, :].T,
                origin='lower',
                cmap='Greys_r',
                vmin=0,
                vmax=1.5)
ax[0, 1].imshow(e[:, :, slz].T, cmap='Greys_r', vmin=0, vmax=vmax)
ax[1, 1].imshow(e[slx, :, :].T,
                origin='lower',
                cmap='Greys_r',
                vmin=0,
                vmax=vmax)
ax[0, 2].imshow(a[:, :, slz].T, cmap='Greys_r', vmin=0, vmax=vmax)
ax[1, 2].imshow(a[slx, :, :].T,
                origin='lower',
                cmap='Greys_r',
                vmin=0,
                vmax=vmax)
ax[0, 3].imshow(b[:, :, slz].T, cmap='Greys_r', vmin=0, vmax=vmax)
ax[1, 3].imshow(b[slx, :, :].T,
                origin='lower',
                cmap='Greys_r',
                vmin=0,
                vmax=vmax)

ax[0, 4].imshow(c[:, :, slz].T, cmap='Greys_r', vmin=0, vmax=1)
ax[1, 4].imshow(c[slx, :, :].T, origin='lower', cmap='Greys_r', vmin=0, vmax=1)

for axx in ax.ravel():
    axx.set_axis_off()

fig.tight_layout()
fig.show()