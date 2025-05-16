import warnings
import cv2
from scipy.ndimage import convolve

from build_spectrogram import *
from analysis_window import *
from gammatone_filter import *
from create_intensity_binary_mask import *
from equal_loudness_correct import create_equal_loudness_values


def create_ref_patches(ref_sig_phaseogram, patch_size):
    """
    Create patches from the reference signal phaseogram to test the test signal against.

    Args:
        ref_sig_phaseogram (np.ndarray): Reference signal phaseogram.
        patch_size (int): Size of each patch.

    Returns:
        list: List of patches.
        np.ndarray: Array containing the x-offsets for the corresponding patch start indices.
    """
    ref_patch_indices = np.arange(patch_size // 2, ref_sig_phaseogram.shape[1] - patch_size, patch_size).astype(int)
    patches = []

    for start_index in ref_patch_indices:
        end_index = start_index + patch_size
        patch = ref_sig_phaseogram[:, start_index - 1: end_index - 1]

        n_rows, n_cols = patch.shape

        # Creating coordinates for interpolation
        xi, yi = np.meshgrid(np.linspace(0, n_cols - 1, n_cols), np.arange(n_rows))
        map_x = xi.astype(np.float32)
        map_y = yi.astype(np.float32)

        # Perform interpolation
        patch_interp = cv2.remap(patch,
                                 map_x,
                                 map_y,
                                 interpolation=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REFLECT_101
                                 )

        patches.append(patch_interp)

    return patches, ref_patch_indices


def create_test_patches(test_patch_frame_indices, test_sig_phaseogram, ref_patches):
    """
    Create the test patches from the test indices for comparison
    with the reference patches.

    Args:
        test_patch_frame_indices (ndarray): Indices of frames in the test spectrogram to be used as patches.
        test_sig_phaseogram: (ndarray): The test spectrogram.
        ref_patches (list of ndarray): List of reference patches.

    Returns:
        list of ndarray: The test patches.
    """
    num_patches = test_patch_frame_indices.shape[0]
    test_patches = []

    for idx in range(num_patches):
        frame_idx = test_patch_frame_indices[idx]
        ref_patch = ref_patches[idx]
        patch_width = ref_patch.shape[1]

        start_col = max(1, frame_idx)
        end_col = start_col + patch_width - 1
        if end_col <= test_sig_phaseogram.shape[1]:
            test_patch = test_sig_phaseogram[:, start_col - 1:end_col]
            test_patches.append(test_patch)

    return test_patches

def calc_ref_test_similarity(ref_patches, test_patches, mask_patches, dynamic_range):
    """
    Calculate the NSIM between all reference and test patches.

    Parameters:
        ref_patches (list of ndarray): List of reference patches.
        test_patches (list of ndarray): List of test patches.
        mask_patches (list of ndarray): List of mask patches.
        dynamic_range (float): Dynamic range of the phaseogram.

    Returns:
        tuple: A tuple containing:
            - mean_patch_nsims (ndarray): Vector of NSIMs representing quality of patches.
            - neurogram_map_patches (list of ndarray): The neurogram maps generated from the NSIM calculations.
    """
    num_patches = len(ref_patches)
    mean_patch_nsims = np.zeros(num_patches)
    neurogram_map_patches = []

    for patch_index in range(num_patches):
        ref_patch = ref_patches[patch_index]
        test_patch = test_patches[patch_index]

        neurogram_map = nsim(ref_patch, test_patch, dynamic_range)

        neurogram_map = neurogram_640_to_32(neurogram_map)

        neurogram_map = mask_patches[patch_index] * neurogram_map

        mean_of_freq_band_sim_means = calc_mean_nsim(neurogram_map)

        mean_patch_nsims[patch_index] = mean_of_freq_band_sim_means
        neurogram_map_patches.append(neurogram_map)

    return mean_patch_nsims, neurogram_map_patches


def nsim(ref_patch, test_patch, dynamic_range):
    """
    Compute the NSIM map between a reference patch and a test patch.

    Parameters:
        ref_patch (ndarray): Reference patch.
        test_patch (ndarray): test patch.
        dynamic_range (float): Dynamic range of the phaseogram.

    Returns:
        ndarray: The similarity map between the reference and test patches.
    """

    # Set window size for NSIM comparison
    window = np.array([[0.0113, 0.0838, 0.0113],
                       [0.0838, 0.6193, 0.0838],
                       [0.0113, 0.0838, 0.0113]])
    window /= np.sum(window)

    K = [0.01, 0.03]
    C1 = (K[0] * dynamic_range) ** 2
    C3 = ((K[1] * dynamic_range) ** 2) / 2

    ref_patch = ref_patch.astype(np.float64)
    test_patch = test_patch.astype(np.float64)

    mu_r = convolve(ref_patch, window, mode='reflect')
    mu_d = convolve(test_patch, window, mode='reflect')

    mu_r_mu_d = mu_r * mu_d
    mu_r_sq = mu_r ** 2
    mu_d_sq = mu_d ** 2

    sigma_r_sq = convolve(ref_patch ** 2, window, mode='reflect') - mu_r_sq
    sigma_d_sq = convolve(test_patch ** 2, window, mode='reflect') - mu_d_sq
    sigma_r_d = convolve(ref_patch * test_patch, window, mode='reflect') - mu_r_mu_d

    intensity = (2 * mu_r_mu_d + C1) / (mu_r_sq + mu_d_sq + C1)
    structure = (sigma_r_d + C3) / (np.sqrt(sigma_r_sq * sigma_d_sq) + C3)

    similarity_map = intensity * structure

    return similarity_map


def neurogram_640_to_32(neurogram_map_640):
    """
    Convert a 640-bin neurogram map to a 32-bin neurogram map.

    Args:
        neurogram_map_640 (ndarray): The original 640-bin neurogram map.

    Returns:
        ndarray: The converted 32-bin neurogram map.
    """
    bins_per_band = [2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 6, 6, 7, 8, 9, 10,
                     11, 14, 14, 17, 19, 22, 25, 28, 32, 36, 41, 47, 53, 60, 68, 78]

    lower_limits = np.cumsum([0] + bins_per_band[:-1])
    upper_limits = np.cumsum(bins_per_band)

    num_bands = len(bins_per_band)
    neurogram_map_32 = np.zeros((num_bands, neurogram_map_640.shape[1]))

    for i in range(neurogram_map_640.shape[1]):
        for j in range(num_bands):
            neurogram_map_32[j, i] = np.mean(neurogram_map_640[lower_limits[j]:upper_limits[j], i])

    return neurogram_map_32


def calc_mean_nsim(neurogram_map):
    """
    Calculate the mean NSIM from a neurogram map.

    Args:
        neurogram_map (ndarray): The neurogram map containing similarity measures.

    Returns:
        float: The mean NSIM value.
    """

    # Compute the mean NSIM for each frequency band
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        band_mean_similarities = np.nanmean(neurogram_map, axis=1)

    # Compute the overall mean NSIM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_nsim = np.nanmean(band_mean_similarities)

    return mean_nsim

def ignore_high_freq_bands(intensity_mask, freq_band_threshold):
    """
    Modify the intensity binary mask by setting frequency bands above (or equal)
    to the 'freq_band_threshold' level to NaN values.

    Args:
    intensity_mask (np.ndarray): The input intensity binary mask.
    freq_band_threshold (int): The frequency band threshold above which to ignore bands.

    Returns:
    np.ndarray: The modified intensity binary mask with specified frequency bands set to NaN.
    """

    n_bins, n_frames = intensity_mask.shape
    nan_values = np.full(n_frames, np.nan)

    for band in range(freq_band_threshold - 1, n_bins):
        intensity_mask[band, :] = nan_values

    return intensity_mask


def calc_vnsim(ref_sig, test_sig, sample_rate, intensity_threshold=-180, elc=0, ignore_freq_bands=0):
    """
    Calculate the VNSIM between reference and test signals.

    Args:
        ref_sig (np.ndarray): Reference signal.
        test_sig (np.ndarray): test signal.
        sample_rate (int): Sample rate of the audio signals.
        intensity_threshold (int): Threshold for intensity binary masking.
        elc (int): Equal loudness contour adjustment parameter:
            0 - no elc
            1 - elc by boosting low and high frequencies
            2 - elc by attenuating low and high frequencies
        ignore_freq_bands (int): ignoring high frequency bands (0:32):
            0 - all 32 frequency bands are taken into account
            k - k-th to 32 frequency bands are ignored in calculations

    Returns:
        float: The VNSIM score.
    """

    # constants:
    PATCH_SIZE = 30
    dynamic_range = 2 * np.pi

    # STEP 1: get gammatonegrams and phaseograms for both signals
    # create analysis window and filter bank instances
    analysis_window = AnalysisWindow()
    filter_bank = GammatoneFilter()

    # calculate time spaces for ref_sig and test_sig
    ref_sig_time_spaces = analysis_window.calc_time_spaces(ref_sig)
    test_sig_time_spaces = analysis_window.calc_time_spaces(test_sig)

    # build gammatonegrams for both signals
    ref_sig_gtgram = build_gammatonegram(ref_sig, sample_rate, filter_bank, ref_sig_time_spaces)
    test_sig_gtgram = build_gammatonegram(test_sig, sample_rate, filter_bank, test_sig_time_spaces)

    # build phaseograms for both signals
    ref_sig_phaseogram = build_phaseogram(ref_sig, analysis_window)
    test_sig_phaseogram = build_phaseogram(test_sig, analysis_window)

    # STEP 2: Apply equal loudness values if required
    if elc > 0:
        el_values = create_equal_loudness_values(ref_sig_gtgram.shape)

        if elc == 1:  # boost low and high frequencies
            ref_sig_gtgram += el_values
            test_sig_gtgram += el_values
        elif elc == 2:  # attenuate low and high frequencies
            ref_sig_gtgram -= el_values
            test_sig_gtgram -= el_values

    # STEP 3: calculate intensity binary masks and patches
    intensity_mask_ref = create_intensity_binary_mask(ref_sig_gtgram, intensity_threshold)
    intensity_mask_test = create_intensity_binary_mask(test_sig_gtgram, intensity_threshold)

    combined_intensity_mask = intensity_mask_ref + intensity_mask_test
    combined_intensity_mask = create_intensity_binary_mask_nan(combined_intensity_mask)

    if ignore_freq_bands > 1:
        combined_intensity_mask = ignore_high_freq_bands(combined_intensity_mask, ignore_freq_bands)

    mask_patches, _ = create_ref_patches(combined_intensity_mask, PATCH_SIZE)

    # STEP 4: create reference signal and test signal patches
    ref_patches, ref_patch_indices = create_ref_patches(ref_sig_phaseogram, PATCH_SIZE)
    test_patches = create_test_patches(ref_patch_indices, test_sig_phaseogram, ref_patches)

    # STEP 5: calculate patch similarity
    patch_similarities, similarity_maps = calc_ref_test_similarity(ref_patches, test_patches, mask_patches, dynamic_range)

    # STEP 6: calculate VNSIM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        vnsim = np.nanmean(patch_similarities)

    return vnsim
