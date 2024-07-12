
from skimage.metrics import structural_similarity as ssim


def ssim_score_5D(reconstruction, y):
    shape = y.shape
    ssim_score = 0
    for t in range(shape[-1]):
        for s in range(shape[-2]):
            ssim_score += ssim(reconstruction[:, :, :, s, t],
                               y[:, :, :, s, t], data_range=1.0)
    return ssim_score / (shape[-1] * shape[-2])
