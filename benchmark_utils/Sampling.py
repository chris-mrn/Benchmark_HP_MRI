import numpy as np
from benchmark_utils.Mask import power_3D_density_mask


# Create a 4D spatial spectral mask
def sampled_kspace5D_mask(kspace):
    nx, ny, nz, ns, nt = kspace.shape
    sampled_kspace_5D = np.zeros((nx, ny, nz, ns, nt), dtype=np.complex128)
    for t in range(nt):
        for s in range(ns):
            mask = power_3D_density_mask((nx, ny, nz), 12)
            sampled_kspace_5D[:, :, :, s, t] = mask * kspace[:, :, :, s, t]

    return sampled_kspace_5D
