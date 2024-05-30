import numpy as np


def power_density_mask(kspace_shape, m):
    power_mask = np.zeros(kspace_shape, dtype=float)
    center_x, center_y = power_mask.shape[0] // 2, power_mask.shape[1] // 2
    max_radius = np.sqrt(center_x ** 2 + center_y ** 2)

    for i in range(power_mask.shape[0]):
        for j in range(power_mask.shape[1]):
            radius = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            norm_radius = 1 + radius/max_radius
            # give a 1 with probability

            power_mask[i, j] = np.random.uniform(0, 1) < 2/(1+norm_radius**m)

    return power_mask
