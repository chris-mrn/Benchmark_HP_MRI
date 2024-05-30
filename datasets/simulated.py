from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from brainweb_dl import get_mri
    from benchmark_utils.mask import power_density_mask
    import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "simulated"

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

        mri_data = get_mri(sub_id=43, contrast="T1")
        mri_data = mri_data[::-1, ...]

        image = mri_data[90, :, :]
        image = image/256
        mask = power_density_mask(image.shape, 8)

        kspace = np.fft.fft2(image, norm='ortho')
        kspace = np.fft.fftshift(kspace)

        undersampled_kspace = mask*kspace

        # The dictionary defines the keyword arguments for `Objective.set_data`

        # I want to return a dictionary with the undersampled image regarding
        # a specific mask and the ground truth image.

        # i only want images no kspace data
        # X will be a list of images with the zero filling reconstruction

        return dict(X=undersampled_kspace, y=image)
