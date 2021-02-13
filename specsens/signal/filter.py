import numpy as np
from scipy import signal

def fir_bandpass(x, f_min, f_max, f_sample, n_taps=64):
    return fir_bandpass_helper(x,
                               f_center=(f_min + f_max) * .5,
                               f_width=np.abs(f_max - f_min),
                               f_sample=f_sample,
                               n_taps=n_taps)


def fir_bandpass_helper(x, f_center, f_width, f_sample, n_taps=255):
    fir = signal.firwin(n_taps, f_width * .5, fs=f_sample, window='hamming')
    n = np.arange(0, n_taps)
    fir = fir * np.exp(1.j * 2 * np.pi * n * f_center * 1. / f_sample)
    return signal.fftconvolve(x, fir, mode='valid')
