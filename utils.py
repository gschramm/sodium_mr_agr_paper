from __future__ import annotations

import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
from typing import Union, Sequence

from pymirc.image_operations import zoom3d

import SimpleITK as sitk

def read_single_tpi_gradient_file(gradient_file: str,
                                  gamma_by_2pi: float = 1126.2,
                                  num_header_elements: int = 6):

    header = np.fromfile(gradient_file,
                         dtype=np.int16,
                         offset=0,
                         count=num_header_elements)

    # number of cones
    num_cones = int(header[0])
    # number of points in a single readout
    num_points = int(header[1])

    # time sampling step in seconds
    dt = float(header[2]) * (1e-6)

    # maximum gradient strength in G/cm corresponds to max short value (2**15 - 1 = 32767
    max_gradient = float(header[3]) / 100

    # number of readouts per cone
    num_readouts_per_cone = np.fromfile(gradient_file,
                                        dtype=np.int16,
                                        offset=num_header_elements * 2,
                                        count=num_cones)

    gradient_array = np.fromfile(gradient_file,
                                 dtype=np.int16,
                                 offset=(num_header_elements + num_cones) * 2,
                                 count=num_cones * num_points).reshape(
                                     num_cones, num_points)

    # calculate k_array in (1/cm)
    k_array = np.cumsum(
        gradient_array,
        axis=1) * dt * gamma_by_2pi * max_gradient / (2**15 - 1)

    return k_array, header, num_readouts_per_cone


def read_tpi_gradient_files(file_base: str,
                            x_suffix: str = 'x.grdb',
                            y_suffix: str = 'y.grdb',
                            z_suffix: str = 'z.grdb',
                            **kwargs):

    kx, header, num_readouts_per_cone = read_single_tpi_gradient_file(
        f'{file_base}.{x_suffix}', **kwargs)
    ky, header, num_readouts_per_cone = read_single_tpi_gradient_file(
        f'{file_base}.{y_suffix}', **kwargs)
    kz, header, num_readouts_per_cone = read_single_tpi_gradient_file(
        f'{file_base}.{z_suffix}', **kwargs)

    kx_rotated = np.zeros((num_readouts_per_cone.sum(), kx.shape[1]))
    ky_rotated = np.zeros((num_readouts_per_cone.sum(), ky.shape[1]))
    kz_rotated = np.zeros((num_readouts_per_cone.sum(), kz.shape[1]))

    num_readouts_cumsum = np.cumsum(
        np.concatenate(([0], num_readouts_per_cone)))

    # start angle of first readout in each cone
    phi0s = np.linspace(0, 2 * np.pi, kx.shape[0], endpoint=False)

    for i_cone in range(header[0]):
        num_readouts = num_readouts_per_cone[i_cone]

        phis = np.linspace(phi0s[i_cone],
                           2 * np.pi + phi0s[i_cone],
                           num_readouts,
                           endpoint=False)

        for ir in range(num_readouts):
            kx_rotated[ir + num_readouts_cumsum[i_cone], :] = np.cos(
                phis[ir]) * kx[i_cone, :] - np.sin(phis[ir]) * ky[i_cone, :]
            ky_rotated[ir + num_readouts_cumsum[i_cone], :] = np.sin(
                phis[ir]) * kx[i_cone, :] + np.cos(phis[ir]) * ky[i_cone, :]
            kz_rotated[ir + num_readouts_cumsum[i_cone], :] = kz[i_cone, :]

    return kx_rotated, ky_rotated, kz_rotated, header, num_readouts_per_cone


def show_tpi_readout(kx,
                     ky,
                     kz,
                     header,
                     num_readouts_per_cone,
                     start_cone=0,
                     end_cone=None,
                     cone_step=2,
                     readout_step=6,
                     step=20):
    num_cones = header[0]

    if end_cone is None:
        end_cone = num_cones

    cone_numbers = np.arange(start_cone, end_cone, cone_step)

    # cumulative sum of readouts per cone
    rpc_cumsum = np.concatenate(([0], num_readouts_per_cone.cumsum()))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for ic in cone_numbers:
        ax.scatter3D(
            kx[rpc_cumsum[ic]:rpc_cumsum[ic + 1]:readout_step, ::step],
            ky[rpc_cumsum[ic]:rpc_cumsum[ic + 1]:readout_step, ::step],
            kz[rpc_cumsum[ic]:rpc_cumsum[ic + 1]:readout_step, ::step],
            s=0.5)

    ax.set_xlim(kx.min(), kx.max())
    ax.set_ylim(ky.min(), ky.max())
    ax.set_zlim(kz.min(), kz.max())
    fig.tight_layout()
    fig.show()


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


def setup_brainweb_phantom(
    simulation_matrix_size: int,
    phantom_data_path: Path,
    field_of_view_cm: float = 22.,
    csf_na_concentration: float = 3.0,
    gm_na_concentration: float = 1.5,
    wm_na_concentration: float = 1.0,
    other_na_concentration: float = 0.75,
    T2long_ms_csf: float = 50.,
    T2long_ms_gm: float = 15.,
    T2long_ms_wm: float = 18.,
    T2long_ms_other: float = 15.,
    T2short_ms_csf: float = 50.,
    T2short_ms_gm: float = 8.,
    T2short_ms_wm: float = 9.,
    T2short_ms_other: float = 8.,
    add_anatomical_mismatch: bool = False,
    add_T2star_bias: bool = False):

    simulation_voxel_size_mm: float = 10 * field_of_view_cm / simulation_matrix_size

    # setup the phantom on a high resolution grid (0.5^3mm) first
    label_nii = nib.load(phantom_data_path / 'subject54_crisp_v.nii.gz')
    label_nii = nib.as_closest_canonical(label_nii)

    # pad to 220mm FOV
    lab_voxelsize = label_nii.header['pixdim'][1]
    lab = label_nii.get_fdata()
    pad_size_220 = ((220 - np.array(lab.shape) * lab_voxelsize) /
                    lab_voxelsize / 2).astype(int)
    pad_size_220 = ((pad_size_220[0], pad_size_220[0]),
                    (pad_size_220[1], pad_size_220[1]), (pad_size_220[2],
                                                         pad_size_220[2]))
    lab = np.pad(lab, pad_size_220, 'constant')

    # CSF = 1, GM = 2, WM = 3
    csf_inds = np.where(lab == 1)
    gm_inds = np.where(lab == 2)
    wm_inds = np.where(lab == 3)
    other_inds = np.where(lab >= 4)
    skull_inds = np.where(lab == 7)

    # calculate eye masks
    x = np.arange(lab.shape[0])
    X, Y, Z = np.meshgrid(x, x, x)
    R1 = np.sqrt((X - 368)**2 + (Y - 143)**2 + (Z - 97)**2)
    R2 = np.sqrt((X - 368)**2 + (Y - 291)**2 + (Z - 97)**2)
    eye1_inds = np.where((R1 < 25))
    eye2_inds = np.where((R2 < 25))

    # set up array for trans. magnetization
    img = np.zeros(lab.shape, dtype=np.float32)
    img[csf_inds] = csf_na_concentration
    img[gm_inds] = gm_na_concentration
    img[wm_inds] = wm_na_concentration
    img[other_inds] = other_na_concentration
    img[skull_inds] = 0.1
    img[eye1_inds] = csf_na_concentration
    img[eye2_inds] = csf_na_concentration

    # set up array for Gamma (ratio between 2nd and 1st echo)
    T2short_ms = np.full(lab.shape,
                         0.5 * np.finfo(np.float32).max,
                         dtype=np.float32)
    T2short_ms[csf_inds] = T2short_ms_csf
    T2short_ms[gm_inds] = T2short_ms_gm
    T2short_ms[wm_inds] = T2short_ms_wm
    T2short_ms[other_inds] = T2short_ms_other
    T2short_ms[eye1_inds] = T2short_ms_csf
    T2short_ms[eye2_inds] = T2short_ms_csf

    T2long_ms = np.full(lab.shape,
                        0.5 * np.finfo(np.float32).max,
                        dtype=np.float32)
    T2long_ms[csf_inds] = T2long_ms_csf
    T2long_ms[gm_inds] = T2long_ms_gm
    T2long_ms[wm_inds] = T2long_ms_wm
    T2long_ms[other_inds] = T2long_ms_other
    T2long_ms[eye1_inds] = T2long_ms_csf
    T2long_ms[eye2_inds] = T2long_ms_csf

    # read the T1
    t1_nii = nib.load(phantom_data_path / 'subject54_t1w_p4_resampled.nii.gz')
    t1_nii = nib.as_closest_canonical(t1_nii)
    t1 = np.pad(t1_nii.get_fdata(), pad_size_220, 'constant')

    # add eye contrast
    t1[eye1_inds] *= 0.5
    t1[eye2_inds] *= 0.5

    # add mismatches
    if add_anatomical_mismatch:
        R1 = np.sqrt((X - 329)**2 + (Y - 165)**2 + (Z - 200)**2)
        inds1 = np.where((R1 < 10))
        img[inds1] = gm_na_concentration
        #R2 = np.sqrt((X - 327)**2 + (Y - 262)**2 + (Z - 200)**2)
        #inds2 = np.where((R2 < 10))
        #t1[inds2] = 0

    # add bias field on T2* times
    if add_T2star_bias:
        T2starbias = np.arctan((Z - 155) / 10) / (2 * np.pi) + 0.75
        T2short_ms *= T2starbias
        T2long_ms *= T2starbias

    # extrapolate the all images to the voxel size we need for the data simulation
    img_extrapolated = zoom3d(img, lab_voxelsize / simulation_voxel_size_mm)
    T2short_ms_extrapolated = zoom3d(T2short_ms,
                                     lab_voxelsize / simulation_voxel_size_mm)
    T2long_ms_extrapolated = zoom3d(T2long_ms,
                                    lab_voxelsize / simulation_voxel_size_mm)
    t1_extrapolated = zoom3d(t1, lab_voxelsize / simulation_voxel_size_mm)

    return img_extrapolated, t1_extrapolated, T2short_ms_extrapolated, T2long_ms_extrapolated


def numpy_volume_to_sitk_image(vol, voxel_size, origin):
    image = sitk.GetImageFromArray(np.swapaxes(vol, 0, 2))
    image.SetSpacing(voxel_size.astype(np.float64))
    image.SetOrigin(origin.astype(np.float64))

    return image


def sitk_image_to_numpy_volume(image):
    vol = np.swapaxes(sitk.GetArrayFromImage(image), 0, 2)

    return vol


def align_images(fixed_image: np.ndarray,
                 moving_image: np.ndarray,
                 fixed_voxsize: Sequence[float] = (1., 1., 1.),
                 fixed_origin: Sequence[float] = (0., 0., 0.),
                 moving_voxsize: Sequence[float] = (1., 1., 1.),
                 moving_origin: Sequence[float] = (0., 0., 0.),
                 final_transform: Union[sitk.Transform, None] = None,
                 verbose: bool = True):

    fixed_sitk_image = numpy_volume_to_sitk_image(
        fixed_image.astype(np.float32), fixed_voxsize, fixed_origin)
    moving_sitk_image = numpy_volume_to_sitk_image(
        moving_image.astype(np.float32), moving_voxsize, moving_origin)

    if final_transform is None:
        # Initial Alignment
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_sitk_image, moving_sitk_image, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY)

        # Registration
        registration_method = sitk.ImageRegistrationMethod()

        # Similarity metric settings.
        registration_method.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(
            registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.1)

        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings.
        registration_method.SetOptimizerAsGradientDescentLineSearch(
            learningRate=0.2,
            numberOfIterations=400,
            convergenceMinimumValue=1e-7,
            convergenceWindowSize=10)

        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel(
            smoothingSigmas=[2, 1, 0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell multiple times.
        registration_method.SetInitialTransform(initial_transform,
                                                inPlace=False)

        final_transform = registration_method.Execute(
            sitk.Cast(fixed_sitk_image, sitk.sitkFloat32),
            sitk.Cast(moving_sitk_image, sitk.sitkFloat32))

        # Post registration analysis
        if verbose:
            print(
                f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}"
            )
            print(
                f"Final metric value: {registration_method.GetMetricValue()}")
            print(f"Final parameters: {final_transform.GetParameters()}")

    moving_sitk_image_resampled = sitk.Resample(moving_sitk_image,
                                                fixed_sitk_image,
                                                final_transform,
                                                sitk.sitkLinear, 0.0,
                                                moving_sitk_image.GetPixelID())

    moving_image_aligned = sitk_image_to_numpy_volume(
        moving_sitk_image_resampled)

    return moving_image_aligned, final_transform


def read_GE_ak_wav(fname: str):
    """
    read GE waveforms stored to external file using Stanford format

    Parameters
    ----------
    fname : str

    Returns
    -------
    grad: np.ndarray
        Gradient waveforms in (T/m)
        shape (#pts/interleave,#interleaves,#groups)
        with: #groups = 2 for 2d-imaging, =3 for 3d-imaging
    bw: float
        full bandwidth (opuser0 = 2d3*oprbw)           [Hz]
    fov: float
        field-of-view (old=nucleus; new=1H equivalent) [m]
    desc: np.ndarray
        description string (256 chars)
    N: dict
         N.gpts   # input gradient pts/interleave
         N.groups # groups
         N.intl   # interleaves
         N.params # parameters
    params : np.ndarray
        header file parameters (scanner units)
        grad_type fov N.intl gmax N.gpts gdt N.kpts kdt 0 0 0
    """
    N = {}
    offset = 0

    desc = np.fromfile(fname, dtype=np.int8, offset=offset, count=256)
    offset += desc.size * desc.itemsize

    N["gpts"] = np.fromfile(fname,
                            dtype=np.dtype('>u2'),
                            offset=offset,
                            count=1)[0]
    offset += N["gpts"].size * N["gpts"].itemsize

    N["groups"] = np.fromfile(fname,
                              dtype=np.dtype('>u2'),
                              offset=offset,
                              count=1)[0]
    offset += N["groups"].size * N["groups"].itemsize

    N["intl"] = np.fromfile(fname,
                            dtype=np.dtype('>u2'),
                            offset=offset,
                            count=N["groups"])
    offset += N["intl"].size * N["intl"].itemsize

    N["params"] = np.fromfile(fname,
                              dtype=np.dtype('>u2'),
                              offset=256 + 4 + N["groups"] * 2,
                              count=1)[0]
    offset += N["params"].size * N["params"].itemsize

    params = np.fromfile(fname,
                         dtype=np.dtype('>f8'),
                         offset=offset,
                         count=N["params"])
    offset += params.size * params.itemsize

    wave = np.fromfile(fname, dtype=np.dtype('>i2'), offset=offset)
    offset += wave.size * wave.itemsize

    grad = np.swapaxes(wave.reshape((N["groups"], N["intl"][0], N["gpts"])), 0,
                       2)

    # set stop bit to 0
    grad[-1, ...] = 0

    # scale gradients to SI units (T/m)
    grad = (grad / 32767) * (params[3] / 100)

    # bandwidth in (Hz)
    bw = 1e6 / params[7]

    # (proton) field of view in (m)
    fov = params[1] / 100

    return grad, bw, fov, desc, N, params
