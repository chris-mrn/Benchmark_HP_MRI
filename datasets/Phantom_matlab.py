from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.Phantom_generator import Phantom_5D_HP_MRI
    import numpy as np
    from benchmark_utils.Sampling import sampled_5D_matlab_waves


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Phantom_matlab"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {}

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = ["pip:brainweb_dl"]

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        phantom_generator = Phantom_5D_HP_MRI(sub_id=45,
                                              contrast="T1",
                                              size=(48, 48, 24),
                                              acquisition_time=120,
                                              time_points=5,
                                              spectral_length=32)

        phantom = phantom_generator.make_5D_HP_MRI_phantom()
        image = phantom

        kspace = np.fft.fftshift(np.fft.fftn(image, norm='ortho'),
                                 axes=(0, 1, 2))

        # Matlab parameters for the GE 3T scanner
        points_per_wave = 8192
        number_of_waves = 3500
        MM = 20
        RNx = 48
        RNy = RNx
        RNz = RNx/2
        fOV = RNx*10e-2
        DPnkDW = 512e-6
        kDW = 30
        maxDG_Tpms = 1

        undersampled_kspace = sampled_5D_matlab_waves(kspace,
                                                      points_per_wave,
                                                      number_of_waves,
                                                      MM,
                                                      RNx,
                                                      RNy,
                                                      RNz,
                                                      fOV,
                                                      DPnkDW,
                                                      kDW,
                                                      maxDG_Tpms)

        X = undersampled_kspace
        y = image

        return dict(X=X, y=y)
