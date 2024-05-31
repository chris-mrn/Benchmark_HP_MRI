import numpy as np
from benchmark_utils.mask import power_density_mask


def undersampling_kspace(image, mask):
    mask = power_density_mask(image.shape, 8)

    kspace = np.fft.fft2(image, norm='ortho')
    kspace = np.fft.fftshift(kspace)

    undersampled_kspace = mask*kspace

    return undersampled_kspace
