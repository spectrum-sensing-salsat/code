import numpy as np
from scipy import stats

from specsens import util


def pfa(thr, n, m):
    return stats.norm.sf((thr - 1.) / np.sqrt((n + m) / (n * m)))


def pd(noise_power, signal_power, thr, n, m, dB=True, num_bands=1):
    if dB:
        snr = util.dB_to_factor_power(
            signal_power) / util.dB_to_factor_power(noise_power)
    else:
        snr = signal_power / noise_power
    # noise energy is distributed over all bands, but signal is not
    # therefore increase SNR by number of bands
    snr *= num_bands
    return stats.norm.sf((thr / (1 + snr) - 1) / np.sqrt((n + m) / (n * m)))


def thr(pfa, n, m):
    return stats.norm.isf(pfa) * np.sqrt((n + m) / (n * m)) + 1


def roc(noise_power, signal_power, pfa, n, dB=True, num_bands=1):
    if dB:
        snr = util.dB_to_factor_power(
            signal_power) / util.dB_to_factor_power(noise_power)
    else:
        snr = signal_power / noise_power
    # noise energy is distributed over all bands, but signal is not
    # therefore increase SNR by number of bands
    snr *= num_bands
    return stats.norm.sf((stats.norm.isf(pfa) + 1./n)/(1.+snr)-1./n)
