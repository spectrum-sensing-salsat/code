import numpy as np
from scipy import signal
import pywt


def expand_edges(x, percent):
    if percent > 1.:
        percent_len = x.size * percent // 100
    else:
        percent_len = int(x.size * percent)
    x = np.append(x, np.repeat(x[-1], percent_len))
    x = np.insert(x, 0, np.repeat(x[0], percent_len))
    return x, percent_len


def edge_detector(spec, freq, scale=4, min_height=0.1, min_freq=1e4):
    sig, m = expand_edges(
        spec, 0.1)  # expand the spectrum at the edges by 0.1 percent
    # powers of 2 used for the multiscale product
    pows2 = np.array(list(map(lambda x: 2.**x, np.arange(1, scale + 1))))
    coef, freqs = pywt.cwt(data=sig,  # apply wavelet transformations with first order derivative gaussian wavelet
                           scales=pows2,
                           wavelet='gaus1',
                           sampling_period=1,
                           method='fft')
    # do the multiscale product and shrink edges
    prod = np.prod(np.abs(coef), axis=0)[m:-m]
    prod = prod / np.max(prod)  # normalize
    df = freq[1] - freq[0]
    peak, info = signal.find_peaks(
        x=prod, height=min_height, distance=min_freq / df)  # do the peak finding
    peak_freq = np.array(
        list(map(lambda x: int((x - len(spec) // 2) * df), peak)))
    return prod, peak, peak_freq
