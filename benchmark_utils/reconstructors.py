import pyproximal
import numpy as np
import pylops


# create a class inspired my skleanr API where .fit() is the iteration of
# the reconstruction
# and .transform() is the final reconstruction (gtpd)
class MRI_Reconstructor:
    def __init__(self,
                 prior=pyproximal.proximal.L21(ndim=2),
                 prior_coeff=1):

        self.prior = prior
        self.prior_coeff = prior_coeff

    def reconstruct(self, undersampled_kspace, n_iter=100):

        shape = undersampled_kspace.shape
        size = undersampled_kspace.size

        d = undersampled_kspace

        mask = np.abs(d) > 0

        d = d.ravel()
        d = d[mask.ravel() == 1]

        Fourier_operator = pylops.signalprocessing.FFT2D(dims=shape,
                                                         fftshift_after=True)
        Mask_operator = pylops.Restriction(size,
                                           np.where(mask.ravel() == 1)[0],
                                           dtype=np.complex128)
        Dop = Mask_operator*Fourier_operator

        with pylops.disabled_ndarray_multiplication():
            sigma = 0.04
            prior = self.prior
            data_fidelity = pyproximal.proximal.L2(Op=Dop,
                                                   b=d,
                                                   niter=50,
                                                   warm=True)
            Gop = sigma * pylops.Gradient(dims=shape,
                                          edge=True,
                                          kind='forward',
                                          dtype=np.complex128)

            L = sigma ** 2 * 8
            tau = .99 / np.sqrt(L)
            mu = .99 / np.sqrt(L)
            gtpd = pyproximal.optimization.primaldual.PrimalDual(
                                data_fidelity,
                                prior,
                                Gop,
                                x0=np.zeros(size),
                                tau=tau,
                                mu=mu,
                                theta=1.,
                                niter=n_iter,
                                show=True)
        reconstruction = np.real(gtpd.reshape(shape))

        return reconstruction
