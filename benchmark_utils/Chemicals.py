import numpy as np
from scipy.signal import find_peaks
from joblib import Parallel, delayed


# Optimized find_amplitudes_chemicals function
def find_amplitudes_chemicals(spectrum, n_peaks=5, threshold=10):
    peaks, _ = find_peaks(np.abs(spectrum), threshold=threshold)
    peak_amplitudes = np.abs(spectrum[peaks])
    top_n_peaks_indices = np.argsort(peak_amplitudes)[-n_peaks:]
    top_n_peaks = peak_amplitudes[top_n_peaks_indices]

    if len(top_n_peaks) < n_peaks:
        return np.pad(top_n_peaks, (0, n_peaks - len(top_n_peaks)), 'constant')

    return top_n_peaks


# Optimized make_chemicals_images function with parallel processing
def process_voxel(recon, x, y, z, n_chemicals, threshold, n_time):
    chemical_amplitudes = np.zeros((n_chemicals, n_time), dtype=np.complex128)
    for t in range(n_time):
        amplitudes = find_amplitudes_chemicals(recon[x, y, z, :, t],
                                               n_peaks=n_chemicals,
                                               threshold=threshold)
        chemical_amplitudes[:, t] = amplitudes
    return (x, y, z, chemical_amplitudes)


def make_chemicals_images(recon, n_chemicals=5, threshold=0):
    nx, ny, nz, _, n_time = recon.shape
    chemicals_images_time = np.zeros((nx, ny, nz, n_chemicals, n_time),
                                     dtype=np.complex128)

    results = Parallel(n_jobs=-1)(delayed(process_voxel)(recon,
                                                         x,
                                                         y,
                                                         z,
                                                         n_chemicals,
                                                         threshold,
                                                         n_time)
                                  for x in range(nx) for y in range(ny)
                                  for z in range(nz))

    for x, y, z, chemical_amplitudes in results:
        chemicals_images_time[x, y, z, :, :] = chemical_amplitudes

    return chemicals_images_time
