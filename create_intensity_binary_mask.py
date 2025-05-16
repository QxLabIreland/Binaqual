import numpy as np


def create_intensity_binary_mask(gammatonegram, intensity_threshold):
    """
    Creates an Intensity Binary Mask from the gammatonegram for a given intensity level threshold.
    
    Args:
    gammatonegram (ndarray): The input gammatonegram.
    intensity_threshold (int): The intensity level threshold.
    
    Returns:
    ndarray: The resulting intensity binary mask.
    """
    n_bins, n_frames = gammatonegram.shape
    intensity_binary_mask = np.zeros((n_bins, n_frames))

    for i in range(n_bins):
        for j in range(n_frames):
            if gammatonegram[i, j] >= intensity_threshold:
                intensity_binary_mask[i, j] = 1
            else:
                intensity_binary_mask[i, j] = 0

    return intensity_binary_mask


def create_intensity_binary_mask_nan(gammatonegram):
    """
    Creates an Intensity Binary Mask with NaN values from the gammatonegram.
    
    Args:
    gammatonegram (np.ndarray): The input matrix.

    Returns:
    np.ndarray: The intensity binary mask with NaN values.
    """
    intensity_binary_mask = np.zeros_like(gammatonegram)
    n_bins, n_frames = gammatonegram.shape

    for i in range(n_bins):
        for j in range(n_frames):
            if gammatonegram[i, j] >= 1:
                intensity_binary_mask[i, j] = 1
            else:
                intensity_binary_mask[i, j] = np.nan

    return intensity_binary_mask
