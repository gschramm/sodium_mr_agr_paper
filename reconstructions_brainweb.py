"""
    data_root_dir / tpi_gradients / ak_grad56.wav
    data_root_dir / brainweb54 / 'subject54_t1w_p4_resampled.nii.gz'
    data_root_dir / brainweb54 / 'subject54_crisp_v.nii.gz'
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import cupy as cp
import cupyx.scipy.ndimage as ndimage
import sigpy
from pymirc.image_operations import zoom3d

from utils import setup_brainweb_phantom, read_GE_ak_wav
from utils_sigpy import NUFFTT2starDualEchoModel, projected_gradient_operator

#--------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_dir', help = 'directory containing the input data')
parser.add_argument('--max_num_iter', type=int, default=2000)
parser.add_argument('--num_iter_r', type=int, default=100)
parser.add_argument('--noise_level', type=float, default=1e-2)
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--sigma', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--beta_anatomical', type=float, default=3e-3)
parser.add_argument('--beta_non_anatomical', type=float, default=1e-1)
parser.add_argument('--beta_r', type=float, default=3e-1)
parser.add_argument('--regularization_norm_anatomical',
                    type=str,
                    default='L1',
                    choices=['L1', 'L2'])
parser.add_argument('--regularization_norm_non_anatomical',
                    type=str,
                    default='L2',
                    choices=['L1', 'L2'])
parser.add_argument('--eta', type=float, default=0.005)
args = parser.parse_args()

data_root_dir = args.data_root_dir
max_num_iter = args.max_num_iter
num_iter_r = args.num_iter_r
noise_level = args.noise_level
no_decay = args.no_decay
sigma = args.sigma
seed = args.seed
beta_anatomical = args.beta_anatomical
beta_non_anatomical = args.beta_non_anatomical
beta_r = args.beta_r
regularization_norm_anatomical = args.regularization_norm_anatomical
regularization_norm_non_anatomical = args.regularization_norm_non_anatomical
eta = args.eta

phantom = 'brainweb'

cp.random.seed(seed)

#---------------------------------------------------------------
# fixed parameters

# image shape for data simulation
sim_shape = (256, 256, 256)
num_sim_chunks = 128

# image shape for iterative recons
ishape = (128, 128, 128)
# grid shape for IFFTs
grid_shape = (64, 64, 64)

field_of_view_cm: float = 22.

# 10us sampling time
acq_sampling_time_ms: float = 0.016

# echo times in ms
echo_time_1_ms = 0.455
echo_time_2_ms = 5.

# signal fraction decaying with short T2* time
short_fraction = 0.6

time_bin_width_ms: float = 0.25

odir = Path(
    data_root_dir
) / 'run_brainweb' / f'{phantom}_nodecay_{no_decay}_i_{max_num_iter:04}_{num_iter_r:04}_nl_{noise_level:.1E}_s_{seed:03}'
odir.mkdir(exist_ok=True, parents=True)

with open(odir / 'config.json', 'w') as f:
    json.dump(vars(args), f, indent=4)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# setup the image

if no_decay:
    decay_suffix = '_no_decay'
    T2long_ms_csf: float = 1e7
    T2long_ms_gm: float = 1e7
    T2long_ms_wm: float = 1e7
    T2short_ms_csf: float = 1e7
    T2short_ms_gm: float = 1e7
    T2short_ms_wm: float = 1e7
else:
    decay_suffix = ''
    T2long_ms_csf: float = 50.
    T2long_ms_gm: float = 20.
    T2long_ms_wm: float = 18.
    T2short_ms_csf: float = 50.
    T2short_ms_gm: float = 3.
    T2short_ms_wm: float = 3.

field_of_view_cm: float = 22.

# (1) setup the brainweb phantom with the given simulation matrix size
x, t1_image, T2short_ms, T2long_ms = setup_brainweb_phantom(
    sim_shape[0],
    Path(data_root_dir) / 'brainweb54',
    field_of_view_cm=field_of_view_cm,
    T2long_ms_csf=T2long_ms_csf,
    T2long_ms_gm=T2long_ms_gm,
    T2long_ms_wm=T2long_ms_wm,
    T2short_ms_csf=T2short_ms_csf,
    T2short_ms_gm=T2short_ms_gm,
    T2short_ms_wm=T2short_ms_wm,
    csf_na_concentration=1.5,
    gm_na_concentration=0.6,
    wm_na_concentration=0.4,
    other_na_concentration=0.3,
    add_anatomical_mismatch=True,
    add_T2star_bias=False)

# move image to GPU
x = cp.asarray(x.astype(np.complex128))

true_ratio_image_short = cp.array(
    np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2short_ms))
true_ratio_image_long = cp.array(
    np.exp(-(echo_time_2_ms - echo_time_1_ms) / T2long_ms))

cp.save(odir / 'na_gt.npy', x)
cp.save(odir / 't1.npy', t1_image)
cp.save(odir / 'true_ratio_short.npy', true_ratio_image_short)
cp.save(odir / 'true_ratio_long.npy', true_ratio_image_long)

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# data simulation block
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------

# time sampling step in micro seconds
dt_us = acq_sampling_time_ms * 1000
# gamma by 2pi in MHz/T
gamma_by_2pi_MHz_T: float = 11.262

# kmax for 64 matrix (edge)
kmax64_1_cm = 1 / (2 * field_of_view_cm / 64)

# read gradients in T/m with shape (num_samples_per_readout, num_readouts, 3)
grads_T_m, bw, fov, desc, N, params = read_GE_ak_wav(
    Path(data_root_dir) / 'tpi_gradients/ak_grad56.wav')

# k values in 1/cm with shape (num_samples_per_readout, num_readouts, 3)
k_1_cm = 0.01 * np.cumsum(grads_T_m, axis=0) * dt_us * gamma_by_2pi_MHz_T

k_1_cm_abs = np.linalg.norm(k_1_cm, axis=2)
print(f'readout kmax .: {k_1_cm_abs.max():.2f} 1/cm')
print(f'64 kmax      .: {kmax64_1_cm:.2f} 1/cm')

data_echo_1 = []
data_echo_2 = []

for i_chunk, k_inds in enumerate(
        np.array_split(np.arange(k_1_cm.shape[0]), num_sim_chunks)):
    print('simulating data chunk', i_chunk)

    data_model = NUFFTT2starDualEchoModel(
        sim_shape,
        k_1_cm[k_inds, ...],
        field_of_view_cm=field_of_view_cm,
        acq_sampling_time_ms=acq_sampling_time_ms,
        time_bin_width_ms=time_bin_width_ms,
        echo_time_1_ms=echo_time_1_ms + k_inds[0] *
        acq_sampling_time_ms,  # account of acq. offset time of every chunk
        echo_time_2_ms=echo_time_2_ms + k_inds[0] *
        acq_sampling_time_ms)  # account of acq. offset time of every chunk

    data_operator_1_short, data_operator_2_short = data_model.get_operators_w_decay_model(
        true_ratio_image_short)
    data_operator_1_long, data_operator_2_long = data_model.get_operators_w_decay_model(
        true_ratio_image_long)

    #--------------------------------------------------------------------------
    # simulate noise-free data
    data_echo_1.append(short_fraction * data_operator_1_short(x) +
                       (1 - short_fraction) * data_operator_1_long(x))
    data_echo_2.append(short_fraction * data_operator_2_short(x) +
                       (1 - short_fraction) * data_operator_2_long(x))

    del data_operator_1_short
    del data_operator_2_short
    del data_operator_1_long
    del data_operator_2_long
    del data_model

data_echo_1 = cp.concatenate(data_echo_1)
data_echo_2 = cp.concatenate(data_echo_2)

# scale data such that max of DC component is 2
data_scale = 2.0 / float(cp.abs(data_echo_1).max())

data_echo_1 *= data_scale
data_echo_2 *= data_scale

# add noise to the data
nl = noise_level * cp.abs(data_echo_1.max())
data_echo_1 += nl * (cp.random.randn(*data_echo_1.shape) +
                     1j * cp.random.randn(*data_echo_1.shape))
data_echo_2 += nl * (cp.random.randn(*data_echo_2.shape) +
                     1j * cp.random.randn(*data_echo_2.shape))

d1 = data_echo_1.reshape(k_1_cm.shape[:-1])
d2 = data_echo_2.reshape(k_1_cm.shape[:-1])

# print info related to SNR
for i in np.linspace(0, d1.shape[1], 10, endpoint=False).astype(int):
    print(
        f'{i:04} {float(cp.abs(d1[:, i]).max() / cp.abs(d1[-100:, i]).std()):.2f}'
    )

#--------------------------------------------------------------------
#--------------------------------------------------------------------
# adjoint NUFFT recon
#--------------------------------------------------------------------
#--------------------------------------------------------------------

# setup the density compensation weights
abs_k = np.linalg.norm(k_1_cm, axis=-1)
abs_k_twist = abs_k[114, 0]

dcf = cp.asarray(np.clip(abs_k**2, None, abs_k_twist**2)).ravel()

ifft1 = sigpy.nufft_adjoint(data_echo_1 * dcf,
                            cp.array(k_1_cm).reshape(-1, 3) * field_of_view_cm,
                            grid_shape)

ifft2 = sigpy.nufft_adjoint(data_echo_2 * dcf,
                            cp.array(k_1_cm).reshape(-1, 3) * field_of_view_cm,
                            grid_shape)

# scaling factor for iffts
ifft_scale = 0.4 / 0.035
ifft1 *= ifft_scale
ifft2 *= ifft_scale

# interpolate to recons (128) grid
ifft1 = ndimage.zoom(ifft1,
                     ishape[0] / grid_shape[0],
                     order=1,
                     prefilter=False)
ifft2 = ndimage.zoom(ifft2,
                     ishape[0] / grid_shape[0],
                     order=1,
                     prefilter=False)

ifft1_sm = ndimage.gaussian_filter(ifft1, 2.)
ifft2_sm = ndimage.gaussian_filter(ifft2, 2.)

cp.save(odir / 'adjoint_ifft_echo_1.npy', ifft1)
cp.save(odir / 'adjoint_ifft_echo_2.npy', ifft2)

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

# calculate an effective scaling factor for the reconstructed images
img_scale = (
    (sim_shape[0] / ishape[0])**(3 / 2)) * (data_scale / acq_model.scale)

# save all scaling factors to a file
with open(odir / 'scaling_factors.json', 'w') as f:
    json.dump(
        {
            'image_scale': img_scale,
            'data_scale': data_scale,
            'acq_model_scale': acq_model.scale
        }, f)

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

outfile1 = odir / f'recon_echo_1_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}_{max_num_iter}.npz'

if not outfile1.exists():
    max_eig_wo_decay = sigpy.app.MaxEig(A.H * A,
                                        dtype=cp.complex128,
                                        device=data_echo_1.device,
                                        max_iter=30).run()
    alg1 = sigpy.alg.PrimalDualHybridGradient(proxfc=proxfc1,
                                              proxg=sigpy.prox.NoOp(A.ishape),
                                              A=A,
                                              AH=A.H,
                                              x=deepcopy(img_scale * ifft1_sm),
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

    cp.savez(outfile1, x=alg1.x, u=u1, max_eig=max_eig_wo_decay)
    recon_echo_1_wo_decay_model = alg1.x
else:
    d1 = cp.load(outfile1)
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
                                              x=deepcopy(img_scale * ifft2_sm),
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
# projected gradient operator that we need for DTV

t1_image /= np.percentile(t1_image, 99.9)

prior_image = cp.asarray(zoom3d(t1_image, ishape[0] / sim_shape[0]))
PG = projected_gradient_operator(ishape, prior_image, eta=eta)

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
step = 0.3
acq_model.dual_echo_data = cp.concatenate([data_echo_1, data_echo_2])

agr_both_echos_w_decay_model = deepcopy(agr_echo_1_wo_decay_model)

proxfcb = sigpy.prox.Stack([
    sigpy.prox.L2Reg(data_echo_1.shape, 1, y=-data_echo_1),
    sigpy.prox.L2Reg(data_echo_2.shape, 1, y=-data_echo_2),
    sigpy.prox.Conj(proxg)
])

num_outer = max_num_iter // num_iter_r

outfileb = odir / f'agr_both_echo_w_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_r:.1E}_{max_num_iter}_{num_iter_r}.npz'
outfile_r = odir / f'est_ratio_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_r:.1E}_{max_num_iter}_{num_iter_r}.npy'

if not outfileb.exists():
    for i_outer in range(num_outer):
        print(f'outer iteration {i_outer+1} / {num_outer}')
        # regenerate recon operators with updated estimated ratio for T2* decay modeling
        recon_operator_1, recon_operator_2 = acq_model.get_operators_w_decay_model(
            est_ratio)
        A = sigpy.linop.Vstack([recon_operator_1, recon_operator_2, PG])

        if i_outer == 0:
            ub = cp.concatenate([
                u1[:data_echo_1.size], u1[:data_echo_1.size],
                u1[data_echo_1.size:]
            ])

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

        agr_both_echos_w_decay_model = algb.x

        del A
        del recon_operator_1
        del recon_operator_2

        #---------------------------------------------------------------------------------------
        # optimize the ratio image using gradient descent on data fidelity + prior
        #---------------------------------------------------------------------------------------

        acq_model.x = agr_both_echos_w_decay_model

        print('updating ratio image')
        # projected gradient descent on ratio image
        for i in range(num_iter_r):
            print(f'{(i+1):04} / {num_iter_r:04}', end='\r')

            # gradient based on data fidelity
            grad_df = acq_model.data_fidelity_gradient_r(est_ratio)
            # gradient of beta_r * ||PG(r)||_2^2
            grad_prior = beta_r * PG.H(PG(est_ratio))
            est_ratio = cp.clip(est_ratio - step * (grad_df + grad_prior),
                                1e-2, 1)

    cp.savez(outfileb, x=algb.x, u=ub, max_eig=max_eig_w_decay)
    cp.save(outfile_r, est_ratio)
else:
    db = cp.load(outfileb)
    agr_both_echos_w_decay_model = db['x']
    ub = db['u']
    max_eig_w_decay = float(db['max_eig'])
    est_ratio = cp.load(outfile_r)