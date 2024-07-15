import numpy as np
from benchmark_utils.Mask import power_3D_density_mask
import matlab.engine
from numba import njit, prange


# TN, WN, MW, RN, FOV, DPnkDW, kDW, maxDG_Tpms
def sampled_5D_matlab_waves(kspace_5D,
                            points_per_wave,
                            number_of_waves,
                            MM,
                            RNx,
                            RNy,
                            RNz,
                            fOV,
                            DPnkDW,
                            kDW,
                            maxDG_Tpms):
    eng = matlab.engine.start_matlab()
    eng.addpath('/Users/christophermarouani/Desktop/Code_vesco_2D')
    enc_g = eng.write_5D_LFRwaves_1H(points_per_wave,
                                     number_of_waves,
                                     MM,
                                     RNx,
                                     RNy,
                                     RNz,
                                     fOV,
                                     DPnkDW,
                                     kDW,
                                     maxDG_Tpms,
                                     0)
    enc_g = np.array(enc_g)
    enc_g = np.swapaxes(enc_g, 0, 1)
    sampled_kspace_5D = make_new_5D_kspace_sampled(kspace_5D, enc_g)
    return sampled_kspace_5D


# Create a 5D spatial spectral temporal mask
def sampled_kspace5D_mask(kspace):
    nx, ny, nz, ns, nt = kspace.shape
    sampled_kspace_5D = np.zeros((nx, ny, nz, ns, nt), dtype=np.complex128)
    for t in range(nt):
        for s in range(ns):
            mask = power_3D_density_mask((nx, ny, nz), 10)
            sampled_kspace_5D[:, :, :, s, t] = mask * kspace[:, :, :, s, t]

    return sampled_kspace_5D


# Function to get k-space locations from k-space gradients
@njit
def one_wave_kspace_loc(one_wave_kgrad):
    return np.floor(one_wave_kgrad).astype(np.int64)


@njit(parallel=True)
def make_new_5D_kspace_sampled(kspace, kgrad):
    n_waves, n_points_per_wave, _ = kgrad.shape
    image_shape = np.array(kspace.shape[:3])
    ns = kspace.shape[3]
    nt = kspace.shape[4]
    center_offset = image_shape // 2
    sampled_kspace = np.zeros_like(kspace)
    kspace_locs = (one_wave_kspace_loc(kgrad) + center_offset) % image_shape
    # % should be optional if the wave are adapted to the FOV

    n_points_per_spec = n_points_per_wave // ns
    if n_points_per_spec == 0:
        raise ValueError("You need to have more points per waveform \
                         than the spectral dimension")

    n_waves_per_time_point = n_waves // nt

    for t in prange(nt):
        for i in range(n_waves_per_time_point):
            wave_number = i + t * n_waves_per_time_point
            for g_point in range(n_points_per_wave):
                kx, ky, kz = kspace_locs[wave_number, g_point]
                s = int(ns * g_point / n_points_per_wave)
                sampled_kspace[kx, ky, kz, s, t] = kspace[kx, ky, kz, s, t]
    return sampled_kspace
