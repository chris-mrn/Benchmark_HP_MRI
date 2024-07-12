import pyproximal
import numpy as np
import pylops


class MRI_Reconstructor:
    """
    MRI_Reconstructor class inspired my skleanr API where .fit()
    is the iteration of the reconstruction
    and .transform() is the final reconstruction (gtpd)
    """
    def __init__(self,
                 n_dim,
                 L=1,
                 prior=0,
                 prior_domain='None',
                 prior_coeff=0.04):

        self.n_dim = n_dim
        self.prior = prior
        self.prior_coeff = prior_coeff
        self.prior_domain = prior_domain
        self.L = L

    def reconstruct(self, undersampled_kspace, n_iter=100):

        shape = undersampled_kspace.shape
        size = undersampled_kspace.size

        d = undersampled_kspace

        mask = np.abs(d) > 0

        d = d.ravel()
        d = d[mask.ravel() == 1]

        if self.n_dim == 1:

            Fourier_operator = pylops.signalprocessing.FFT(dims=shape,
                                                           norm='ortho')

        elif self.n_dim == 2:

            Fourier_operator = pylops.signalprocessing.FFT2D(dims=shape,
                                                             norm='ortho')

        else:
            Fourier_operator = pylops.signalprocessing.FFTND(dims=shape,
                                                             axes=(-5,
                                                                   -4,
                                                                   -3,
                                                                   -2,
                                                                   -1),
                                                             norm='ortho')

        Mask_operator = pylops.Restriction(size,
                                           np.where(mask.ravel() == 1)[0],
                                           dtype=np.complex128)

        Dop = Mask_operator * Fourier_operator

        with pylops.disabled_ndarray_multiplication():

            sigma = self.prior_coeff

            data_fidelity = pyproximal.proximal.L2(Op=Dop,
                                                   b=d,
                                                   niter=50)
            if self.prior_domain == 'None':
                self.prior_domain = pylops.Identity(size)

            # Gop = sigma*pylops.signalprocessing.FFT(dims=shape)

            Gop = sigma * self.prior_domain

            L = self.L
            tau = .99 / np.sqrt(L)
            mu = .99 / np.sqrt(L)
            gtpd = pyproximal.optimization.primaldual.PrimalDual(
                                data_fidelity,
                                self.prior,
                                Gop,
                                x0=np.zeros(size).ravel(),
                                tau=tau,
                                mu=mu,
                                theta=1.0,
                                niter=n_iter,
                                show=False)

        reconstruction = np.abs(gtpd.reshape(shape))

        return reconstruction
