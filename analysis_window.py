import numpy as np
from scipy.signal.windows import hamming


class AnalysisWindow:
    def __init__(self, sample_rate=48000, overlap=0.5):
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.size = self.calc_window_size()
        self.data = self.apply_hamming_window()
        self.window_overlap = int(self.size * self.overlap)

    def calc_window_size(self):
        """
        Size of the analysis window.

        Args:
            sample_rate (int): Sample rate of the signal.

        Returns:
            int: Size of the analysis window.
        """

        window_size = round((self.sample_rate / 8000) * 256)
        if window_size % 2 != 0:
            window_size -= 1
        return window_size

    def apply_hamming_window(self):
        """
        Build the analysis window data.

        Args:
            window_size (int): Size of the window.

        Returns:
            np.ndarray: The window data.
        """

        return hamming(self.size, sym=False)

    def calc_time_spaces(self, signal):
        """
        Calculate the size of each frame.

        Args:
            signal (np.ndarray): The input signal.

        Returns:
            np.ndarray: The time spaces when frames begin (in seconds).
        """
        analysis_window_len = len(self.data)
        signal_len = len(signal)

        n_col = (signal_len - self.window_overlap) // (analysis_window_len - self.window_overlap)
        colindex = 1 + np.arange(n_col) * (analysis_window_len - self.window_overlap)
        time_spaces = ((colindex - 1) + (analysis_window_len / 2)) / self.sample_rate

        return time_spaces