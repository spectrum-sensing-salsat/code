import numpy as np
from scipy import stats

from specsens import util


def pfa(noise_power, thr, n, dB=True):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
    return stats.norm.sf((thr - n * noise_power) / (noise_power * np.sqrt(n)))


def pd(noise_power, signal_power, thr, n, dB=True, num_bands=1):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
        signal_power = util.dB_to_factor_power(signal_power)
    # noise energy is distributed over all bands, but signal is not
    # therefore increase SNR by number of bands
    signal_power *= num_bands
    return stats.norm.sf((thr - n * (noise_power + signal_power)) /
                         (np.sqrt(n) * (noise_power + signal_power)))


def thr(noise_power, pfa, n, dB=True):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
    return (np.sqrt(n) * stats.norm.isf(pfa) + n) * noise_power


def roc(noise_power, signal_power, pfa, n, dB=True, num_bands=1):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
        signal_power = util.dB_to_factor_power(signal_power)
    # noise energy is distributed over all bands, but signal is not
    # therefore increase SNR by number of bands
    signal_power *= num_bands
    return stats.norm.sf(
        (noise_power * stats.norm.isf(pfa) - signal_power * np.sqrt(n)) /
        (noise_power + signal_power))


def num(noise_power, signal_power, pfa, pd, dB=True, num_bands=1):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
        signal_power = util.dB_to_factor_power(signal_power)
    # noise energy is distributed over all bands, but signal is not
    # therefore increase SNR by number of bands
    signal_power *= num_bands
    return int(
        np.ceil(((noise_power * stats.norm.isf(pfa) -
                  ((noise_power + signal_power) * stats.norm.isf(pd))) /
                 signal_power)**2))
