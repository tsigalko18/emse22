from math import log10, sqrt

import cv2.cv2 as cv2
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


# euclidean distance
def mse(original, compressed) -> float:
    return mean_squared_error(original, compressed)


# color histogram matching
def histogram_matching(original, compressed, method="correlation") -> float:
    hist1 = cv2.calcHist([original], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()

    hist2 = cv2.calcHist([compressed], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    if method is "correlation":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    elif method is "chi-squared":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    elif method is "intersection":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
    elif method is "hellinger":
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    else:
        print("histogram matching method unknown. Use one among [correlation, chi-squared, intersection, hellinger]")
        exit()


# peak signal-to-noise ratio (PSNR)
def psnr(original, compressed) -> float:
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))


# structural similarity index measure (ssim)
def ssim_skimage(original, compressed) -> float:
    return ssim(original, compressed, channel_axis=-1)
