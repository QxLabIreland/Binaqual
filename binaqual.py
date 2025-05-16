import soundfile as sf
import numpy as np
from vnsim import calc_vnsim
from argparse import ArgumentParser
from pathlib import Path
import warnings


def load_and_preprocess_signals(ref_path, test_path):
    """
    Load and pre-process reference and test audio signals.

    Args:
        ref_path (Path): Path to the reference audio file.
        test_path (Path): Path to the test audio file.

    Returns:
        tuple: Processed reference signal, test signal, sample rate, and number of channels.
    """

    # reading reference and test audio files
    ref_sig, sample_rate = sf.read(str(ref_path))
    test_sig, _ = sf.read(str(test_path))

    n_channels_ref = ref_sig.shape[1]
    n_channels_test = test_sig.shape[1]

    # Number of samples to append
    num_zeros = 11520

    # Create an array of zeros with shape (11520, number of channels)
    zeros_ref = np.zeros((num_zeros, n_channels_ref))
    zeros_test = np.zeros((num_zeros, n_channels_test))

    # Concatenate the zeros array with the original signals
    ref_sig = np.vstack((zeros_ref, ref_sig))
    test_sig = np.vstack((zeros_test, test_sig))

    return ref_sig, test_sig, sample_rate, n_channels_test


def calculate_binaqual(ref_path, test_path, intensity_threshold=-180, elc=0, ignore_freq_bands=0):
    """
    Calculate the Binaqual metric for the given audio files.

    Args:
        ref_path (Path): Path to the reference audio file.
        test_path (Path): Path to the test audio file.
        intensity_threshold (int): Intensity threshold for NSIM calculation.
        elc (int): Equal loudness contour adjustment parameter:
            0 - no elc
            1 - elc by boosting low and high frequencies
            2 - elc by attenuating low and high frequencies
        ignore_freq_bands (int): ignoring high frequency bands (0:32):
            0 - all 32 frequency bands are taken into account
            k - k-th to 32 frequency bands are ignored in calculations

    Returns:
        tuple: List of NSIM values, LQ, LS values.
    """

    ref_sig, test_sig, sample_rate, n_channels = load_and_preprocess_signals(ref_path, test_path)

    nsim_values_nan = []

    for i in range(2):
        if i >= n_channels:
            vnsim = np.nan

        else:
            vnsim = calc_vnsim(ref_sig[:, i], test_sig[:, i],
                               sample_rate, intensity_threshold, elc, ignore_freq_bands)

        nsim_values_nan.append(vnsim)
        print(f"vnsim_{i}:", round(vnsim, 6))

    nsim_values = []
    for i in range(2):
        if np.isnan(nsim_values_nan[i]):
            nsim_values.append(0.1)
        else:
            nsim_values.append(nsim_values_nan[i])

    LS = nsim_values[0] * nsim_values[1]

    return nsim_values_nan, LS


def parse_args():
    parser = ArgumentParser("Binaqual")

    parser.add_argument("--ref",
                        type=Path,
                        help="Path to reference audio file.",
                        required=True
                        )

    parser.add_argument("--test",
                        type=Path,
                        help="Path to test audio file.",
                        required=True
                        )

    parser.add_argument("--level",
                        type=float,
                        help="Intensity threshold for NSIM calculation.",
                        required=False
                        )

    parser.add_argument("--elc",
                        type=int,
                        help="",
                        required=False
                        )

    parser.add_argument("--ignorefreqbands",
                        type=int,
                        help="Number of frequency bands to ignore.",
                        required=False)

    return parser.parse_args()


def main():
    args = parse_args()

    ref_path = args.ref
    test_path = args.test
    intensity_threshold = args.level
    elc = args.elc
    ignore_freq_bands = args.ignorefreqbands

    if intensity_threshold is None:
        intensity_threshold = -180

    if elc is None:
        elc = 0

    if ignore_freq_bands is None:
        ignore_freq_bands = 0

    nsim_values, LS = calculate_binaqual(ref_path,
                                         test_path,
                                         intensity_threshold,
                                         elc,
                                         ignore_freq_bands)

    print("")
    print("LS: ", LS)


if __name__ == '__main__':
    main()