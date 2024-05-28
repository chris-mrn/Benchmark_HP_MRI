import pyproximal
import numpy as np
import pylops


def reconstruction(kspace, mask, prior=pyproximal.proximal.L21(ndim=2),
                   n_iter=100, sigma_noise=0.01):

    sampling = mask
    sampling1 = np.fft.fftshift(sampling)

    image = np.fft.ifft2(kspace).real

    gt = image/256

    Fop = pylops.signalprocessing.FFT2D(dims=gt.shape)
    Rop = pylops.Restriction(gt.size, np.where(sampling1.ravel() == 1)[0],
                             dtype=np.complex128)
    Dop = Rop * Fop

    # KK spectrum
    GT = Fop * gt.ravel()
    GT = GT.reshape(gt.shape)

    # Data (Masked KK spectrum)
    d = Dop * gt.ravel()

    # adding noise to d
    real_noise = np.random.normal(0, sigma_noise, d.shape)
    imag_noise = np.random.normal(0, sigma_noise, d.shape)
    d = d + real_noise + 1j * imag_noise

    with pylops.disabled_ndarray_multiplication():
        sigma = 0.04
        prior = prior
        data_fidelity = prior(Op=Dop, b=d.ravel(), niter=50, warm=True)
        Gop = sigma * pylops.Gradient(dims=gt.shape, edge=True, kind='forward',
                                      dtype=np.complex128)

        L = sigma ** 2 * 8
        tau = .99 / np.sqrt(L)
        mu = .99 / np.sqrt(L)
        gtpd = pyproximal.optimization.primaldual.PrimalDual(
            data_fidelity,
            prior,
            Gop,
            x0=np.zeros(gt.size),
            tau=tau,
            mu=mu,
            theta=1.,
            niter=n_iter, show=True)
        gtpd = np.real(gtpd.reshape(gt.shape))

    return gtpd
