from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skimage.metrics import mean_squared_error as mse
    from benchmark_utils.Score import ssim_score_5D
    import numpy as np
    from benchmark_utils.Sampling import sampled_5Dkspace
    from benchmark_utils.Chemicals import make_chemicals_images


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Benchmark_HP_MRI"

    # URL of the main repo for this benchmark.
    url = "https://github.com/chris-mrn/Benchmark_HP_MRI"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        'density': ('power', 'gaussian', 'matlab')
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"
    requirements = ["pip:scikit-image"
                    "pip:brainweb_dl",
                    "pip:numba",
                    ]

    def set_data(self, image):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.image = image
        kspace = np.fft.fftshift(np.fft.fftn(image, norm='ortho'),
                                 axes=(0, 1, 2))
        undersampled_kspace = sampled_5Dkspace(kspace, mask=self.density)
        sparsity = np.sum(undersampled_kspace == 0)/undersampled_kspace.size
        self.sparsity = sparsity

        self.X = undersampled_kspace
        self.y = make_chemicals_images(image, n_chemicals=5, threshold=0)

    def evaluate_result(self, reconstruction):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        # Compute the mean squared error between the true and reconstructed
        # images.
        recon_chemicals = make_chemicals_images(reconstruction,
                                                n_chemicals=5,
                                                threshold=0)
        recon_chemicals = np.abs(recon_chemicals)
        mse_score = mse(recon_chemicals, np.abs(self.y))
        ssim_score = ssim_score_5D(recon_chemicals, np.abs(self.y))
        self.value = mse_score
        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(value=mse_score,
                    mse_score=mse_score,
                    ssim_score=ssim_score,
                    sparsity=self.sparsity)

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        reconstruction = np.zeros(self.y.shape)
        return dict(reconstruction=reconstruction)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(X=self.X)
