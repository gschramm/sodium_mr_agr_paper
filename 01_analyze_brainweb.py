import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, zoom, center_of_mass
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import ImageGrid

sns.set_context('paper')

parser = argparse.ArgumentParser()
parser.add_argument('--data_root_dir', default = './data', help = 'directory containing the input data')
parser.add_argument('--max_num_iter', type=int, default=2000)
parser.add_argument('--num_iter_r', type=int, default=100)
parser.add_argument('--noise_level', type=float, default=1e-2)
parser.add_argument('--phantom',
                    type=str,
                    default='brainweb',
                    choices=['brainweb', 'blob'])
parser.add_argument('--no_decay', action='store_true')
parser.add_argument('--regularization_norm_anatomical',
                    type=str,
                    default='L1',
                    choices=['L1', 'L2'])
parser.add_argument('--regularization_norm_non_anatomical',
                    type=str,
                    default='L2',
                    choices=['L1', 'L2'])
args = parser.parse_args()

data_root_dir = args.data_root_dir
max_num_iter = args.max_num_iter
num_iter_r = args.num_iter_r
noise_level = args.noise_level
phantom = args.phantom
no_decay = args.no_decay
regularization_norm_anatomical = args.regularization_norm_anatomical
regularization_norm_non_anatomical = args.regularization_norm_non_anatomical

beta_rs = [1e-1, 3e-1, 3e-1]
noise_metric = 2

betas_non_anatomical = [1e-2, 3e-2, 1e-1]
betas_anatomical = [3e-4, 1e-3, 3e-3]
sm_sigs = [0.0, 1.8, 3.]
#-----------------------------------------------------------------------

iter_shape = (128, 128, 128)
grid_shape = (64, 64, 64)

field_of_view_cm: float = 22.

odirs = sorted(
    list((Path(data_root_dir) / 'run_brainweb').glob(
        f'{phantom}_nodecay_{no_decay}_i_{max_num_iter:04}_{num_iter_r:04}_nl_{noise_level:.1E}_s_*'
    )))

#-----------------------------------------------------------------------
# load the ground truth
gt = np.abs(np.load(odirs[0] / 'na_gt.npy'))
t1 = np.abs(np.load(odirs[0] / 't1.npy'))
sim_shape = gt.shape

# load the brainweb label array
phantom_data_path: Path = Path(data_root_dir) / 'brainweb54'

label_nii = nib.as_closest_canonical(
    nib.load(phantom_data_path / 'subject54_crisp_v.nii'))

# pad to 220mm FOV
lab_voxelsize = label_nii.header['pixdim'][1]
lab = np.asanyarray(label_nii.dataobj)
pad_size_220 = ((220 - np.array(lab.shape) * lab_voxelsize) / lab_voxelsize /
                2).astype(int)
pad_size_220 = ((pad_size_220[0], pad_size_220[0]),
                (pad_size_220[1], pad_size_220[1]), (pad_size_220[2],
                                                     pad_size_220[2]))
lab = np.pad(lab, pad_size_220, 'constant')

lab = zoom(lab, sim_shape[0] / lab.shape[0], order=0, prefilter=False)

# create a GM mask
gm_mask = (lab == 2).astype(np.uint8)
wm_mask = (lab == 3).astype(np.uint8)
csf_mask = (lab == 1).astype(np.uint8)

# load the aparc parcelation
aparc_nii = nib.as_closest_canonical(
    nib.load(phantom_data_path / 'aparc.DKTatlas+aseg_native.nii.gz'))
aparc = np.pad(np.asanyarray(aparc_nii.dataobj), pad_size_220, 'constant')

aparc = zoom(aparc, sim_shape[0] / aparc.shape[0], order=0, prefilter=False)

roi_inds = OrderedDict()
roi_inds['ventricles'] = np.where(np.isin((aparc * csf_mask), [4, 43]))

# add the eyes ROI
x = np.linspace(0, 440 - 1, lab.shape[0])
X, Y, Z = np.meshgrid(x, x, x)
R1 = np.sqrt((X - 368)**2 + (Y - 143)**2 + (Z - 97)**2)
R2 = np.sqrt((X - 368)**2 + (Y - 291)**2 + (Z - 97)**2)
eye1_inds = np.where((R1 < 25))
eye2_inds = np.where((R2 < 25))

tmp = np.zeros_like(gt, dtype=np.uint8)
tmp[eye1_inds] = 1
tmp[eye2_inds] = 1

roi_inds['eyes'] = np.where(tmp * (np.abs(gt - 1.5) < 0.01))

# add lesion ROI
R1 = np.sqrt((X - 329)**2 + (Y - 165)**2 + (Z - 200)**2)
roi_inds['lesion'] = np.where((R1 < 10) * (np.abs(gt - 0.6) < 0.01))

roi_inds['white matter'] = np.where(np.isin((aparc * wm_mask), [2, 41]))

roi_inds['putamen'] = np.where(np.isin((aparc * gm_mask), [12, 51]))
roi_inds['caudate'] = np.where(np.isin((aparc * gm_mask), [11, 50]))
roi_inds['cerebellum'] = np.where(np.isin((aparc * gm_mask), [8, 47]))
roi_inds['cortical grey matter'] = np.where((aparc * gm_mask) >= 1000)
roi_inds['frontal'] = np.where(np.isin((aparc * gm_mask), [1028, 2028]))
roi_inds['temporal'] = np.where(
    np.isin((aparc * gm_mask), [1009, 1015, 1030, 2009, 2015, 2030]))

iffts_e1 = np.zeros((
    len(odirs),
    len(sm_sigs),
) + sim_shape)
recons_e1_no_decay = np.zeros((
    len(odirs),
    len(betas_non_anatomical),
) + sim_shape)
agrs_e1_no_decay = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

agrs_both_echos_w_decay0 = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

agrs_both_echos_w_decay1 = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

r0s = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

r0 = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

r1 = np.zeros((
    len(odirs),
    len(betas_anatomical),
) + sim_shape)

# calculate the ROI averages
true_means = {}
recon_e1_no_decay_roi_means = {}
agr_e1_no_decay_roi_means = {}
agr_both_echos_w_decay0_roi_means = {}
agr_both_echos_w_decay1_roi_means = {}
iffts_e1_roi_means = {}
T2star0_roi_means = {}
T2star1_roi_means = {}

recon_e1_no_decay_roi_stds = {}
agr_e1_no_decay_roi_stds = {}
agr_both_echos_w_decay0_roi_stds = {}
agr_both_echos_w_decay1_roi_stds = {}
iffts_e1_roi_stds = {}

sl = int(0.4375 * gt.shape[0])

# calculate the true means
for key, inds in roi_inds.items():
    true_means[key] = gt[inds].mean()

# load the image scale factor
with open(odirs[0] / 'scaling_factors.json', 'r') as f:
    image_scale = json.load(f)['image_scale']

for i, odir in enumerate(odirs):
    print('loading IFFT1', odir)
    # adjoint iffts are already scaled to the scale of the ground truth
    # no need to apply the scaling factor
    tmp = zoom(np.load(odir / 'adjoint_ifft_echo_1.npy'),
               sim_shape[0] / iter_shape[0],
               order=1,
               prefilter=False)

    for ib, sig in enumerate(sm_sigs):
        iffts_e1[i, ib, ...] = np.abs(gaussian_filter(tmp, sig))

for key, inds in roi_inds.items():
    iffts_e1_roi_means[key] = np.array([[x[inds].mean() for x in y]
                                        for y in iffts_e1])
iffts_e1_mean = iffts_e1.mean(axis=0)
iffts_e1_std = iffts_e1.std(axis=0)

for key, inds in roi_inds.items():
    iffts_e1_roi_stds[key] = np.array([x[inds].mean() for x in iffts_e1_std])

for i, odir in enumerate(odirs):
    print('loading iterative no decay model', odir)

    # load iterative recons of first echo with non-anatomical prior
    for ib, beta_non_anatomical in enumerate(betas_non_anatomical):
        ofile_e1_no_decay = odir / f'recon_echo_1_no_decay_model_{regularization_norm_non_anatomical}_{beta_non_anatomical:.1E}_{max_num_iter}.npz'
        d = np.load(ofile_e1_no_decay)
        recons_e1_no_decay[i, ib, ...] = zoom(np.abs(d['x'] / image_scale),
                                              sim_shape[0] / iter_shape[0],
                                              order=1,
                                              prefilter=False)

for key, inds in roi_inds.items():
    recon_e1_no_decay_roi_means[key] = np.array([[x[inds].mean() for x in y]
                                                 for y in recons_e1_no_decay])

recons_e1_no_decay_mean = recons_e1_no_decay.mean(axis=0)
recons_e1_no_decay_std = recons_e1_no_decay.std(axis=0)

for key, inds in roi_inds.items():
    recon_e1_no_decay_roi_stds[key] = np.array(
        [x[inds].mean() for x in recons_e1_no_decay_std])

for i, odir in enumerate(odirs):
    print('loading AGR no decay model', odir)
    # load AGR of first echo with out decay model
    for ib, beta_anatomical in enumerate(betas_anatomical):
        ofile_e1_no_decay_agr = odir / f'agr_echo_1_no_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{max_num_iter}.npz'
        d = np.load(ofile_e1_no_decay_agr)
        agrs_e1_no_decay[i, ib, ...] = zoom(np.abs(d['x'] / image_scale),
                                            sim_shape[0] / iter_shape[0],
                                            order=1,
                                            prefilter=False)

for key, inds in roi_inds.items():
    agr_e1_no_decay_roi_means[key] = np.array([[x[inds].mean() for x in y]
                                               for y in agrs_e1_no_decay])

agrs_e1_no_decay_mean = agrs_e1_no_decay.mean(axis=0)
agrs_e1_no_decay_std = agrs_e1_no_decay.std(axis=0)

for i, odir in enumerate(odirs):
    print('loading AGR w decay model', odir)
    # load AGR of borhs echos with decay model
    for ib, beta_anatomical in enumerate(betas_anatomical):
        ofile_both_echos_agr0 = odir / f'agr_both_echo_w_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_rs[0]:.1E}_{max_num_iter}_{num_iter_r}.npz'
        ofile_both_echos_agr1 = odir / f'agr_both_echo_w_decay_model_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_rs[1]:.1E}_{max_num_iter}_{num_iter_r}.npz'

        outfile_r0 = odir / f'est_ratio_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_rs[0]:.1E}_{max_num_iter}_{num_iter_r}.npy'
        outfile_r1 = odir / f'est_ratio_{regularization_norm_anatomical}_{beta_anatomical:.1E}_{beta_rs[1]:.1E}_{max_num_iter}_{num_iter_r}.npy'

        d0 = np.load(ofile_both_echos_agr0)
        d1 = np.load(ofile_both_echos_agr1)

        agrs_both_echos_w_decay0[i, ib,
                                 ...] = zoom(np.abs(d0['x'] / image_scale),
                                             sim_shape[0] / iter_shape[0],
                                             order=1,
                                             prefilter=False)
        agrs_both_echos_w_decay1[i, ib,
                                 ...] = zoom(np.abs(d1['x'] / image_scale),
                                             sim_shape[0] / iter_shape[0],
                                             order=1,
                                             prefilter=False)

        # load ratio images
        r0[i, ib, ...] = zoom(np.load(outfile_r0),
                              sim_shape[0] / iter_shape[0],
                              order=1,
                              prefilter=False)

        r1[i, ib, ...] = zoom(np.load(outfile_r1),
                              sim_shape[0] / iter_shape[0],
                              order=1,
                              prefilter=False)

for key, inds in roi_inds.items():
    agr_both_echos_w_decay0_roi_means[key] = np.array(
        [[x[inds].mean() for x in y] for y in agrs_both_echos_w_decay0])
    agr_both_echos_w_decay1_roi_means[key] = np.array(
        [[x[inds].mean() for x in y] for y in agrs_both_echos_w_decay1])

agrs_both_echos_w_decay0_mean = agrs_both_echos_w_decay0.mean(axis=0)
agrs_both_echos_w_decay1_mean = agrs_both_echos_w_decay1.mean(axis=0)

agrs_both_echos_w_decay0_std = agrs_both_echos_w_decay0.std(axis=0)
agrs_both_echos_w_decay1_std = agrs_both_echos_w_decay1.std(axis=0)

# convert rs to T2star time

# average ratio images
r0_mean = r0.mean(axis=0)
r1_mean = r1.mean(axis=0)

# scale from ratios to T2star times
T2star0_mean = np.zeros_like(r0_mean)
T2star1_mean = np.zeros_like(r1_mean)

T2star0_mean[r0_mean < 1] = -4.545 / np.log(r0_mean[r0_mean < 1])
T2star1_mean[r1_mean < 1] = -4.545 / np.log(r1_mean[r1_mean < 1])

mask = (gt > 0).astype(float)
for i in range(T2star0_mean.shape[0]):
    T2star0_mean[i, ...] *= mask
    T2star1_mean[i, ...] *= mask

for key, inds in roi_inds.items():
    T2star0_roi_means[key] = np.array([x[inds].mean() for x in T2star0_mean])
    T2star1_roi_means[key] = np.array([x[inds].mean() for x in T2star1_mean])

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--- print the biases in different regions --------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

print(f'\n bias AGR')
for i, (roi, vals) in enumerate(agr_e1_no_decay_roi_means.items()):
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    print(roi, y)

print(f'\n bias AGRdm - beta_r {beta_rs[0]:.2e}')
for i, (roi, vals) in enumerate(agr_both_echos_w_decay0_roi_means.items()):
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    print(roi, y)

print(f'\n bias AGRdm - beta_r {beta_rs[1]:.2e}')
for i, (roi, vals) in enumerate(agr_both_echos_w_decay1_roi_means.items()):
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    print(roi, y)

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--- print mean T2star times  ---------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

print(f'\n T2star0 roi means')
print(T2star0_roi_means)

print(f'\n T2star1 roi means')
print(T2star1_roi_means)

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--- RMSE calculation -----------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

# calculate the RMSE in all ROIs
df_rmse = pd.DataFrame({})

for roi, inds in roi_inds.items():
    for ir in range(recons_e1_no_decay.shape[0]):
        for ib in range(recons_e1_no_decay.shape[1]):
            tmp = pd.DataFrame(dict(
                method='CR',
                b=ib + 1,
                r=ir + 1,
                roi=roi,
                RMSE=np.sqrt(
                    np.mean((recons_e1_no_decay[ir, ib, ...][inds] -
                             gt[inds])**2))),
                               index=[0])
            df_rmse = pd.concat((df_rmse, tmp))

    for ir in range(agrs_e1_no_decay.shape[0]):
        for ib in range(agrs_e1_no_decay.shape[1]):
            tmp = pd.DataFrame(dict(
                method='AGR',
                b=ib + 1,
                r=ir + 1,
                roi=roi,
                RMSE=np.sqrt(
                    np.mean(
                        (agrs_e1_no_decay[ir, ib, ...][inds] - gt[inds])**2))),
                               index=[0])
            df_rmse = pd.concat((df_rmse, tmp))

    for ir in range(agrs_both_echos_w_decay0.shape[0]):
        for ib in range(agrs_both_echos_w_decay0.shape[1]):
            tmp = pd.DataFrame(dict(
                method=f'AGRdm br={beta_rs[0]:.1E}',
                b=ib + 1,
                r=ir + 1,
                roi=roi,
                RMSE=np.sqrt(
                    np.mean((agrs_both_echos_w_decay0[ir, ib, ...][inds] -
                             gt[inds])**2))),
                               index=[0])
            df_rmse = pd.concat((df_rmse, tmp))

    for ir in range(agrs_both_echos_w_decay1.shape[0]):
        for ib in range(agrs_both_echos_w_decay1.shape[1]):
            tmp = pd.DataFrame(dict(
                method=f'AGRdm br={beta_rs[1]:.1E}',
                b=ib + 1,
                r=ir + 1,
                roi=roi,
                RMSE=np.sqrt(
                    np.mean((agrs_both_echos_w_decay1[ir, ib, ...][inds] -
                             gt[inds])**2))),
                               index=[0])
            df_rmse = pd.concat((df_rmse, tmp))

    for ir in range(iffts_e1.shape[0]):
        for ib in range(iffts_e1.shape[1]):
            tmp = pd.DataFrame(dict(method='REGR+IFFT',
                                    b=ib + 1,
                                    r=ir + 1,
                                    roi=roi,
                                    RMSE=np.sqrt(
                                        np.mean((iffts_e1[ir, ib, ...][inds] -
                                                 gt[inds])**2))),
                               index=[0])
            df_rmse = pd.concat((df_rmse, tmp))

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--- Plots ----------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# bias noise plots

num_rows = 2
num_cols = int(np.ceil(len(roi_inds) / 2))

size = 1.5
fig = plt.figure(figsize=(size * 2 * num_cols, size * (2 * num_rows + 2)))
grid = plt.GridSpec(2 * num_rows + 2, 2 * num_cols, figure=fig)

for ir, (key, inds) in enumerate(roi_inds.items()):
    roi_mask = np.zeros_like(gt)
    roi_mask[inds] = 1

    com = [int(x) for x in center_of_mass(roi_mask)]
    sl2 = np.argmax(roi_mask.sum((0, 1)))
    sl1 = np.argmax(roi_mask.sum((0, 2)))

    col = ir % num_cols

    if ir < num_cols:
        rrow = 0
    else:
        rrow = -1

    ax0 = fig.add_subplot(grid[rrow, 2 * col])
    ax1 = fig.add_subplot(grid[rrow, 2 * col + 1])

    ax0.imshow(gt[:, :, sl2].T, origin='lower', cmap='Greys_r')
    ax0.contour(roi_mask[:, :, sl2].T, origin='lower', levels=1, cmap='hot')
    ax0.set_axis_off()
    ax1.imshow(gt[:, sl1, :].T, origin='lower', cmap='Greys_r')
    ax1.contour(roi_mask[:, sl1, :].T, origin='lower', levels=1, cmap='hot')
    ax1.set_axis_off()

axs = []

for i, (roi, vals) in enumerate(recon_e1_no_decay_roi_means.items()):
    if noise_metric == 1:
        x = vals.std(0)
    else:
        x = np.array([z[roi_inds[roi]].mean() for z in recons_e1_no_decay_std])

    col = i % num_cols
    row = int(i >= num_cols)
    axs.append(
        fig.add_subplot(grid[(2 * row + 1):(2 * row + 3),
                             (2 * col):(2 * col + 2)]))

    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    axs[i].plot(x, y, 'o-', label='CR')
    axs[i].set_title(roi)

    for j in range(len(x)):
        axs[i].annotate(f'{betas_non_anatomical[j]:.1E}', (x[j], y[j]),
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize='x-small')

for i, (roi, vals) in enumerate(agr_e1_no_decay_roi_means.items()):
    if noise_metric == 1:
        x = vals.std(0)
    else:
        x = np.array([z[roi_inds[roi]].mean() for z in agrs_e1_no_decay_std])

    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    axs[i].plot(x, y, 'o-', label='AGR')

    for j in range(len(x)):
        axs[i].annotate(f'{betas_anatomical[j]:.1E}', (x[j], y[j]),
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize='x-small')

for i, (roi, vals) in enumerate(agr_both_echos_w_decay0_roi_means.items()):
    if noise_metric == 1:
        x = vals.std(0)
    else:
        x = np.array(
            [z[roi_inds[roi]].mean() for z in agrs_both_echos_w_decay0_std])
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    axs[i].plot(x, y, 'o-', label=f'AGRdm, br={beta_rs[0]:.1E}')

    for j in range(len(x)):
        axs[i].annotate(f'{betas_anatomical[j]:.1E}', (x[j], y[j]),
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize='x-small')

for i, (roi, vals) in enumerate(agr_both_echos_w_decay1_roi_means.items()):
    if noise_metric == 1:
        x = vals.std(0)
    else:
        x = np.array(
            [z[roi_inds[roi]].mean() for z in agrs_both_echos_w_decay1_std])
    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    axs[i].plot(x, y, 'o-', label=f'AGRdm, br={beta_rs[1]:.1E}')

for i, (roi, vals) in enumerate(iffts_e1_roi_means.items()):
    if noise_metric == 1:
        x = vals.std(0)
    else:
        x = np.array([z[roi_inds[roi]].mean() for z in iffts_e1_std])

    y = 100 * (vals.mean(0) - true_means[roi]) / true_means[roi]
    axs[i].plot(x, y, 'o-', label='REGR+IFFT')

    for j in range(len(x)):
        axs[i].annotate(f'{betas_anatomical[j]:.1E}', (x[j], y[j]),
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontsize='x-small')

for i, axx in enumerate(axs):
    axx.grid(ls=':')
    axx.axhline(0, color='k')
    axx.set_xlim(0, 0.12)
    axx.set_ylim(-40, 20)

    if i >= num_cols:
        if noise_metric == 1:
            axx.set_xlabel('std.dev. of ROI mean')
        else:
            axx.set_xlabel('ROI averaged std.dev.')

axs[0].set_ylabel('bias of ROI mean [%]')
axs[num_cols].set_ylabel('bias of ROI mean [%]')
axs[-1].legend(loc='lower right', fontsize='small')

fig.tight_layout()
fig.show()

#-------------------------------------------------------------------------------
# RMSE plots
#------------------------------------------------------------------------------------

df_rmse.reset_index(inplace=True)

strip_kwargs = dict(size=2., dodge=True, palette='tab10')
box_kwargs = dict(showfliers=False,
                  showbox=False,
                  palette='tab10',
                  showcaps=False,
                  showmeans=True,
                  meanline=True,
                  meanprops=dict(color='k', ls='-'),
                  medianprops=dict(visible=False),
                  whiskerprops=dict(visible=False))

g = sns.FacetGrid(df_rmse, col="roi", sharey=False, col_wrap=5, height=2)
g.map_dataframe(sns.stripplot, x="b", y="RMSE", hue="method", **strip_kwargs)
g.map_dataframe(sns.boxplot, x="b", y="RMSE", hue="method", **box_kwargs)
g.add_legend()

for axx in g.axes.ravel():
    axx.grid(ls=':')
    axx.set_xlabel('level of regularization')

g.fig.show()

#--------------------------------------------------------------------------------
# recon plots
#------------------------------------------------------------------------------------

vmax_std = 0.2
vmax = 1.75
bmax = 1
inoise = 0

fig3a, ax3a = plt.subplots(3, 3, figsize=(6, 6))
fig3b, ax3b = plt.subplots(3, 3, figsize=(6, 6))
fig3c, ax3c = plt.subplots(3, 3, figsize=(6, 6))
fig3d, ax3d = plt.subplots(3, 3, figsize=(6, 6))

for i in range(3):
    i0 = ax3a[i, 0].imshow(recons_e1_no_decay[inoise, i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax)
    ax3a[i, 0].set_ylabel(f'$\\beta$ = {betas_non_anatomical[i]:.1E}')
    i1 = ax3a[i,
              1].imshow(recons_e1_no_decay_mean[i, ..., sl].T - gt[:, :, sl].T,
                        origin='lower',
                        cmap='seismic',
                        vmin=-bmax,
                        vmax=bmax)
    i2 = ax3a[i, 2].imshow(recons_e1_no_decay_std[i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax_std)
    i3 = ax3b[i, 0].imshow(agrs_e1_no_decay[inoise, i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax)
    ax3b[i, 0].set_ylabel(f'$\\beta_u$ = {betas_anatomical[i]:.1E}')
    i4 = ax3b[i,
              1].imshow(agrs_e1_no_decay_mean[i, ..., sl].T - gt[:, :, sl].T,
                        origin='lower',
                        cmap='seismic',
                        vmin=-bmax,
                        vmax=bmax)
    i5 = ax3b[i, 2].imshow(agrs_e1_no_decay_std[i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax_std)

    i6 = ax3c[i, 0].imshow(agrs_both_echos_w_decay1[inoise, i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax)
    ax3c[i, 0].set_ylabel(f'$\\beta_u$ = {betas_anatomical[i]:.1E}')
    i7 = ax3c[i, 1].imshow(agrs_both_echos_w_decay1_mean[i, ..., sl].T -
                           gt[:, :, sl].T,
                           origin='lower',
                           cmap='seismic',
                           vmin=-bmax,
                           vmax=bmax)
    i8 = ax3c[i, 2].imshow(agrs_both_echos_w_decay1_std[i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax_std)

    i9 = ax3d[i, 0].imshow(iffts_e1[inoise, i, ..., sl].T,
                           origin='lower',
                           cmap='Greys_r',
                           vmin=0,
                           vmax=vmax)
    ax3d[i, 0].set_ylabel(f'FWHM = {(2.35*sm_sigs[i]*(220/128)):.1f}mm')
    i10 = ax3d[i, 1].imshow(iffts_e1_mean[i, ..., sl].T - gt[:, :, sl].T,
                            origin='lower',
                            cmap='seismic',
                            vmin=-bmax,
                            vmax=bmax)
    i11 = ax3d[i, 2].imshow(iffts_e1_std[i, ..., sl].T,
                            origin='lower',
                            cmap='Greys_r',
                            vmin=0,
                            vmax=vmax_std)

ax3a[0, 0].set_title('first noise realization')
ax3a[0, 1].set_title('bias image')
ax3a[0, 2].set_title('std.dev. image')
ax3b[0, 0].set_title('first noise realization')
ax3b[0, 1].set_title('bias image')
ax3b[0, 2].set_title('std.dev. image')
ax3c[0, 0].set_title('first noise realization')
ax3c[0, 1].set_title('bias image')
ax3c[0, 2].set_title('std.dev. image')
ax3d[0, 0].set_title('first noise realization')
ax3d[0, 1].set_title('bias image')
ax3d[0, 2].set_title('std.dev. image')

for axx in ax3a.ravel():
    axx.xaxis.set_visible(False)
    plt.setp(axx.spines.values(), visible=False)
    axx.tick_params(left=False, labelleft=False)

for axx in ax3b.ravel():
    axx.xaxis.set_visible(False)
    plt.setp(axx.spines.values(), visible=False)
    axx.tick_params(left=False, labelleft=False)

for axx in ax3c.ravel():
    axx.xaxis.set_visible(False)
    plt.setp(axx.spines.values(), visible=False)
    axx.tick_params(left=False, labelleft=False)

for axx in ax3d.ravel():
    axx.xaxis.set_visible(False)
    plt.setp(axx.spines.values(), visible=False)
    axx.tick_params(left=False, labelleft=False)

cax0 = fig3a.add_axes([0.05, 0.04, 0.25, 0.01])
fig3a.colorbar(i0, cax=cax0, orientation='horizontal')
cax1 = fig3a.add_axes([0.05 + 0.325, 0.04, 0.25, 0.01])
fig3a.colorbar(i1, cax=cax1, orientation='horizontal')
cax2 = fig3a.add_axes([0.05 + 0.65, 0.04, 0.25, 0.01])
fig3a.colorbar(i2, cax=cax2, orientation='horizontal')

cax3 = fig3b.add_axes([0.05, 0.04, 0.25, 0.01])
fig3b.colorbar(i3, cax=cax3, orientation='horizontal')
cax4 = fig3b.add_axes([0.05 + 0.325, 0.04, 0.25, 0.01])
fig3b.colorbar(i4, cax=cax4, orientation='horizontal')
cax5 = fig3b.add_axes([0.05 + 0.65, 0.04, 0.25, 0.01])
fig3b.colorbar(i5, cax=cax5, orientation='horizontal')

cax6 = fig3c.add_axes([0.05, 0.04, 0.25, 0.01])
fig3c.colorbar(i6, cax=cax6, orientation='horizontal')
cax7 = fig3c.add_axes([0.05 + 0.325, 0.04, 0.25, 0.01])
fig3c.colorbar(i7, cax=cax7, orientation='horizontal')
cax8 = fig3c.add_axes([0.05 + 0.65, 0.04, 0.25, 0.01])
fig3c.colorbar(i8, cax=cax8, orientation='horizontal')

cax9 = fig3d.add_axes([0.05, 0.04, 0.25, 0.01])
fig3d.colorbar(i9, cax=cax9, orientation='horizontal')
cax10 = fig3d.add_axes([0.05 + 0.325, 0.04, 0.25, 0.01])
fig3c.colorbar(i10, cax=cax10, orientation='horizontal')
cax11 = fig3d.add_axes([0.05 + 0.65, 0.04, 0.25, 0.01])
fig3c.colorbar(i11, cax=cax11, orientation='horizontal')

fig3a.tight_layout(rect=(0, 0.03, 1, 1))
fig3b.tight_layout(rect=(0, 0.03, 1, 1))
fig3c.tight_layout(rect=(0, 0.03, 1, 1))
fig3d.tight_layout(rect=(0, 0.03, 1, 1))

fig3a.savefig('conv_sim.png')
fig3b.savefig('agr_wo_decay_sim.png')
fig3c.savefig('agr_w_decay_sim.png')
fig3d.savefig('ifft1_sim.png')

fig3a.show()
fig3b.show()
fig3c.show()
fig3d.show()

# figure for est. T2star plots
kws = dict(cmap='magma', vmin=0, vmax=50, origin='lower')
fig = plt.figure(figsize=(6, 2))

grid = ImageGrid(
    fig,
    111,  # as in plt.subplot(111)
    nrows_ncols=(1, 3),
    axes_pad=0.15,
    share_all=True,
    cbar_location="right",
    cbar_mode="single",
    cbar_size="7%",
    cbar_pad=0.15,
)

# Add data to image grid
im = grid[0].imshow(T2star1_mean[0, ..., sl].T, **kws)
im = grid[1].imshow(T2star1_mean[1, ..., sl].T, **kws)
im = grid[2].imshow(T2star1_mean[2, ..., sl].T, **kws)

# Colorbar
grid[-1].cax.colorbar(im)
grid[-1].cax.toggle_label(True)

for i, ax in enumerate(grid):
    ax.set_axis_off()
    ax.set_title(f'T2* (ms) $\\beta_u$ = {betas_anatomical[i]:.1E}')

fig.tight_layout()
fig.show()