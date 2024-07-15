from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from brainweb_dl import get_mri
    from scipy.ndimage import zoom
    from scipy.integrate import odeint


class Phantom_5D_HP_MRI:
    """
    A class representing a 5D High-Resolution MRI Phantom.

    This class provides methods to create a 5D MRI phantom by simulating the
    spectral data for each voxel over time.
    The phantom is generated based on the provided MRI data, chemical shifts,
    peak widths, intensity ratios, and kinetic model parameters.

    Attributes:
    - sub_id (int): Subject ID for fetching MRI data.
    - contrast (str): Contrast type for fetching MRI data.
    - size (tuple): Size of the resized brain image.
    - acquisition_time (int): Total acquisition time for the simulation.
    - time_points (int): Number of time points for the simulation.
    - spectral_length (int): Length of the spectral data.

    Methods:
    - resize_brain_image(image, new_size): Resize 3D brain image to new size.
    - create_chemical_tumor(spot_locations, tumor_radius, image_shape):
    Create a chemical tumor in the brain image.
    - two_species_model(y, t, k_SP, R1_S, R1_P): Define the two-species
      kinetic model.
    - make_5D_HP_MRI_phantom(): Generate the 5D MRI phantom.

    """

    def __init__(self, sub_id=45, contrast="T1", size=(30, 30, 24),
                 acquisition_time=120, time_points=10, spectral_length=100):
        self.sub_id = sub_id
        self.contrast = contrast
        self.size = size
        self.acquisition_time = acquisition_time
        self.time_points = time_points
        self.spectral_length = spectral_length

    def resize_brain_image(self, image, new_size):
        """
        Resize a 3D brain image to a new size.

        Parameters:
        - image: 3D numpy array, original brain image.
        - new_size: tuple of ints, desired new size.

        Returns:
        - resized_image: 3D numpy array, resized brain image of shape new_size.
        """
        zoom_factors = [n / o for n, o in zip(new_size, image.shape)]
        resized_image = zoom(image, zoom_factors, order=3)
        return resized_image

    def create_chemical_tumor(self, spot_locations, tumor_radius, image_shape):
        """
        Create a chemical tumor in the brain image.

        Parameters:
        - spot_locations: dict, dictionary of spot locations for each chemical.
        - tumor_radius: int, radius of the tumor.
        - image_shape: tuple of ints, shape of the brain image.

        Returns:
        - chemical_tumor: 3D numpy array, brain image with the chemical tumor.
        """
        chemical_tumor = np.zeros(image_shape)
        max_radius = np.sqrt(
            image_shape[0] ** 2 +
            image_shape[1] ** 2 +
            image_shape[2] ** 2
        )
        for x in range(image_shape[0]):
            for y in range(image_shape[1]):
                for z in range(image_shape[2]):
                    radius = np.sqrt(
                        (x - spot_locations[0]) ** 2 +
                        (y - spot_locations[1]) ** 2 +
                        (z - spot_locations[2]) ** 2
                    )
                    if radius < tumor_radius:
                        chemical_tumor[x, y, z] = np.exp(
                            (-radius ** 2) / max_radius
                        )
        return chemical_tumor

    def two_species_model(self, y, t, k_SP, R1_S, R1_P):
        """
        Define the two-species kinetic model.

        Parameters:
        - y: list, list of initial concentrations of the two species.
        - t: numpy array, time points for the simulation.
        - k_SP: float, rate constant for the conversion between
        the two species.
        - R1_S: float, relaxation rate of species S.
        - R1_P: float, relaxation rate of species P.

        Returns:
        - list: list of derivatives of the concentrations of the two species.
        """
        M_S, M_P, M_H = y
        dM_S_dt = -R1_S * M_S - k_SP * M_S
        dM_P_dt = k_SP * M_S - R1_P * M_P
        dM_H_dt = k_SP * M_S - R1_P * M_H
        return [dM_S_dt, dM_P_dt, dM_H_dt]

    def make_5D_HP_MRI_phantom(self):
        """
        Generate the 5D MRI phantom.

        Returns:
        - phantom: 5D numpy array, MRI phantom with spectral data
        for each voxel over time.
        """
        # Fetch MRI data using brainweb_dl
        mri_data = get_mri(sub_id=self.sub_id, contrast=self.contrast)
        mri_data = mri_data[::-1, ...]
        # changing the order of the dimensions so that it is (x, y, z)
        mri_data = np.transpose(mri_data, (1, 2, 0))

        # Normalize MRI data to [0, 1]
        min_value = np.min(mri_data)
        max_value = np.max(mri_data)
        mri_data = (mri_data - min_value) / (max_value - min_value)

        # Resize the MRI data
        resized_image = self.resize_brain_image(mri_data, self.size)
        mri_data = resized_image

        # Define chemical shifts (in ppm) and
        # peak widths (standard deviation of Gaussian)
        chemical_shifts = {
            'pyruvate': 170.2,
            'lactate': 183.2,
            'bicarbonate': 160.8,
            'alanine': 176.0,
            'pyrH': 179.0
        }

        self.chemical_shifts = chemical_shifts

        peak_widths = {
            'pyruvate': 0.3,
            'lactate': 0.25,
            'bicarbonate': 0.2,
            'alanine': 0.25,
            'pyrH': 0.25,
        }

        self.peak_widths = peak_widths

        # Ratios of intensities for each chemical
        intensity_ratios = {
            'pyruvate': 1.0,
            'lactate': 1 / 2,
            'bicarbonate': 1 / 4,
            'alanine': 2 / 3,
            'pyrH': 1 / 3
        }

        self.intensity_ratios = intensity_ratios

        # Create spectral data for each voxel
        ppm_scale = np.linspace(155, 190, self.spectral_length)

        self.ppm_scale = ppm_scale

        # Precompute Gaussian peaks for each chemical shift and intensity ratio
        gaussian_peaks = {}
        for chem, shift in chemical_shifts.items():
            peak_width = peak_widths[chem]
            gaussian_peaks[chem] = np.exp(
                -0.5 * ((ppm_scale - shift) / peak_width) ** 2
            )

        # Add circular spots for the tumor
        (_, nx, _) = self.size
        tumor_radius = nx//9

        # Generate random locations for spots near
        # the center of the brain volume for each chemical
        max_x, max_y, max_z = mri_data.shape
        center_x, center_y, center_z = max_x // 2, max_y // 2, max_z // 2

        random_spot_tumor = (
            np.clip(
                np.random.randint(int(3/4*center_x), int(5/4*center_x)),
                0,
                max_x - tumor_radius
            ),
            np.clip(
                np.random.randint(int(3/4*center_y), int(5/4*center_y)),
                0,
                max_y - tumor_radius
            ),
            np.clip(
                np.random.randint(int(3/4*center_z), int(5/4*center_z)),
                0,
                max_z - tumor_radius
            )
        )

        center_tumor = random_spot_tumor
        tumor_intensity = 1.5

        # Define the kinetic model parameters
        k_SP = 0.02
        R1_S = 1/30
        R1_P = 1/20

        # Time points for the simulation
        t = np.linspace(0, self.acquisition_time, self.time_points)

        # Create an empty 5D array to store the spectra
        # for each voxel over time
        brain_shape = mri_data.shape
        phantom = np.zeros(brain_shape +
                           (self.spectral_length, self.time_points))

        # Generate spectra for each voxel over time
        for x in range(brain_shape[0]):
            for y in range(brain_shape[1]):
                for z in range(brain_shape[2]):
                    intensity = mri_data[x, y, z]
                    if intensity > 0:
                        M_S0 = intensity
                        M_P0 = 0
                        M_H0 = 0
                        sol = odeint(self.two_species_model,
                                     [M_S0, M_P0, M_H0],
                                     t,
                                     args=(k_SP, R1_S, R1_P))
                        M_S = sol[:, 0]
                        M_P = sol[:, 1]
                        M_H = sol[:, 2]

                        for time_idx, (m_s, m_p, m_h) in enumerate(
                            zip(M_S, M_P, M_H)
                        ):
                            spectrum = np.zeros(self.spectral_length)
                            for chem, ratio in intensity_ratios.items():
                                radius_center_tumor = np.sqrt(
                                    (center_tumor[0]) ** 2 +
                                    (center_tumor[1]) ** 2 +
                                    (center_tumor[2]) ** 2
                                )
                                radius_to_center = np.sqrt(
                                    (x - center_tumor[0]) ** 2 +
                                    (y - center_tumor[1]) ** 2 +
                                    (z - center_tumor[2]) ** 2
                                )

                                if radius_to_center < tumor_radius:
                                    exp_decay = np.exp(
                                        -radius_to_center /
                                        (2*radius_center_tumor)
                                    )
                                    tumor_intensity = 1/2

                                    if chem == 'pyruvate':
                                        peak = (1 +
                                                (ratio *
                                                 tumor_intensity *
                                                 exp_decay)) * m_s * \
                                                 ratio * gaussian_peaks[chem]

                                        spectrum += peak

                                    elif chem == 'pyrH':
                                        peak = (1 +
                                                (ratio *
                                                 tumor_intensity *
                                                 exp_decay)) * m_p * \
                                                 ratio * gaussian_peaks[chem]

                                        spectrum += peak
                                    else:
                                        peak = (1 +
                                                (ratio *
                                                 tumor_intensity *
                                                 exp_decay)) * m_h * \
                                                 ratio * gaussian_peaks[chem]
                                        spectrum += peak

                                else:
                                    if chem == 'pyruvate':
                                        spectrum += m_s * ratio * \
                                                    gaussian_peaks[chem]
                                    elif chem == 'pyrH':
                                        spectrum += m_p * ratio * \
                                                    gaussian_peaks[chem]
                                    else:
                                        spectrum += m_h * ratio * \
                                                    gaussian_peaks[chem]
                            noise_level = 0.02
                            noise = np.abs(noise_level
                                           * np.max(spectrum)
                                           * np.random.randn(
                                               self.spectral_length))
                            noise = 0
                            spectrum += noise
                            phantom[x, y, z, :, time_idx] = spectrum
        return 1000*phantom
