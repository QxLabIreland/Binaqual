import numpy as np
from scipy.interpolate import interp1d


def iso226(phon):
    """
    Calculate the equal-loudness contour for a given phon level based on ISO 226:2003 standard.

    Args:
        phon (float): The phon level for which the contour is calculated.

    Returns:
        tuple: (SPL values in dB, frequencies in Hz)
    """
    frequencies = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                            1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500])

    af = np.array([0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315,
                   0.301, 0.288, 0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243,
                   0.243, 0.242, 0.242, 0.245, 0.254, 0.271, 0.301])

    Lu = np.array([-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1,
                   -2.0, -1.1, -0.4, 0.0, 0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7,
                   2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1])

    Tf = np.array([78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4,
                   11.4, 8.6, 6.2, 4.4, 3.0, 2.2, 2.4, 3.5, 1.7, -1.3, -4.2,
                   -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3])

    if phon < 0 or phon > 90:
        raise ValueError("Phon value out of bounds!")

    Ln = phon
    Af = 4.47E-3 * (10.0 ** (0.025 * Ln) - 1.15) + (0.4 * 10.0 ** (((Tf + Lu) / 10) - 9)) ** af
    Lp = ((10.0 / af) * np.log10(Af)) - Lu + 94

    return Lp, frequencies


def erbs_space(low_freq=100, high_freq=11025, num_bands=100):
    """
    Compute an array of frequencies uniformly spaced between high_freq and low_freq on an ERB scale.

    Args:
        low_freq (int): The lowest frequency in the range.
        high_freq (int): The highest frequency in the range.
        num_bands (int): The number of frequencies to generate.

    Returns:
        numpy.ndarray: Array of frequencies uniformly spaced on an ERB scale.
    """

    # Glasberg and Moore Parameters
    EarQ = 9.26449
    minBW = 24.7

    erb_scale = np.linspace(1, num_bands, num_bands)
    cf_array = -(EarQ * minBW) + np.exp(
        erb_scale * (-np.log(high_freq + EarQ * minBW) + np.log(low_freq + EarQ * minBW)) / num_bands) * (
                           high_freq + EarQ * minBW)

    return cf_array


def equal_loudness_correct(fmin=50, fmax=16000, phon_value=7.4, num_bands=32):
    """
    Calculate the equal loudness curve and interpolate it over a specified number of bands.

    Args:
        fmin (int): Minimum frequency.
        fmax (int): Maximum frequency.
        phon_value (float): Phon level for the equal loudness contour.
        num_bands (int): Number of frequency bands.

    Returns:
        tuple: (Interpolated threshold values, frequencies)
    """
    iso_values, iso_frequencies = iso226(phon_value)

    extra_pts = int(round(len(iso_frequencies) * fmax / iso_frequencies[-1] - len(iso_frequencies)))
    extra_values = np.logspace(np.log10(iso_values[-1]), np.log10(1.3 * max(iso_values)), extra_pts)
    extra_frequencies = np.linspace(iso_frequencies[-1], fmax, extra_pts)

    equal_loudness_frequencies = np.concatenate((iso_frequencies, extra_frequencies[1:]))
    equal_loudness_curve = np.concatenate((iso_values, extra_values[1:]))

    center_frequencies = erbs_space(fmin, fmax, num_bands)
    interpolated_frequencies = np.round(np.sort(center_frequencies))

    interp_func = interp1d(equal_loudness_frequencies, equal_loudness_curve, kind='linear')
    interpolated_thresholds = interp_func(interpolated_frequencies)

    return interpolated_thresholds, interpolated_frequencies


def create_equal_loudness_values(size_gammatonegram):
    """
    Derive equal loudness values from the equal loudness contour for the gammatonegram frequencies.

    Args:
        size_gammatonegram (tuple): Size of the gammatonegram.

    Returns:
        numpy.ndarray: Equal loudness correction values for the gammatonegram.
    """
    el_values, _ = equal_loudness_correct(50, 16000, 7.4, 32)

    el_correction = np.ones(size_gammatonegram)

    for i in range(32):
        el_correction[i, :] *= el_values[i]

    return el_correction
