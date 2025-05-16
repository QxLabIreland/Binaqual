from dataclasses import dataclass

@dataclass
class GammatoneFilter:
    low_freq: int = 50
    high_freq: int = 16000
    num_bands: int = 32
    use_fft: bool = False
