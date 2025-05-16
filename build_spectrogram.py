import numpy as np
from gammatone.gtgram import gtgram
from gammatone.fftweight import fft_gtgram
from scipy.signal import stft


def build_gammatonegram(signal, sample_rate, filter_bank, time_spaces):
    """
    Build a gammatonegram from an audio signal.

    Args:
        signal (numpy.ndarray): The audio signal.
        sample_rate (int): The sample rate of the audio signal.
        filter_bank (GammatoneFilter): An instance of the GammatoneFilter class.
        time_spaces (numpy.ndarray): Array of time indices for the signal.

    Returns:
        numpy.ndarray: The gammatonegram.
    """
    hop_size = time_spaces[1] - time_spaces[0]

    if filter_bank.use_fft:
        spectrogram = fft_gtgram(
            signal, sample_rate, hop_size, hop_size,
            filter_bank.num_bands, filter_bank.low_freq
        )
    else:
        spectrogram = gtgram(
            signal, sample_rate, hop_size, hop_size,
            filter_bank.num_bands, filter_bank.low_freq
        )

    # Get magnitude
    spectrogram = np.abs(spectrogram)

    # Avoid -inf in dB computation
    spectrogram[spectrogram == 0] = np.finfo(float).eps

    # Convert to dB
    gammatonegram = 10 * np.log10(spectrogram)

    return gammatonegram


def build_phaseogram(signal, analysis_window):
    """
    Build a phaseogram from an audio signal.

    Args:
        signal (numpy.ndarray): The audio signal.
        analysis_window (AnalysisWindow): An instance of the AnalysisWindow class.

    Returns:
        numpy.ndarray: The phaseogram.
    """

    # Align with gammatonegram
    zero_padding = np.zeros(analysis_window.window_overlap)
    padded_signal = np.concatenate([zero_padding, signal])

    _, _, spectrogram = stft(
        padded_signal,
        window=analysis_window.data,
        nperseg=analysis_window.size,
        noverlap=analysis_window.window_overlap,
        nfft=2048
    )

    # Remove 0 Hz and frequencies above gammatone fmax
    spectrogram = spectrogram[:, 1:-1]

    # # Extract phase
    phaseogram = np.angle(spectrogram)
    #
    # # Exclude 0 Hz and high frequencies
    phaseogram = phaseogram[1:641, :]

    return phaseogram