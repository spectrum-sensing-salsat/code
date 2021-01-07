import numpy as np
from scipy import stats

from specsens import util


def pfa(noise_power, thr, n, dB=True):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
    return 1. - stats.chi2.cdf(2. * thr / noise_power, 2. * n)


def pd(noise_power, signal_power, thr, n, dB=True, num_bands=1):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
        signal_power = util.dB_to_factor_power(signal_power)
    # noise energy is distributed over all bands, but signal is not
    # therefore increase SNR by number of bands
    signal_power *= num_bands
    return 1. - stats.chi2.cdf(2. * thr / (noise_power + signal_power), 2. * n)


def thr(noise_power, pfa, n, dB=True):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
    return stats.chi2.ppf(1. - pfa, 2. * n) * noise_power / 2.


def roc(noise_power, signal_power, pfa, n, dB=True, num_bands=1):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
        signal_power = util.dB_to_factor_power(signal_power)
    # noise energy is distributed over all bands, but signal is not
    # therefore increase SNR by number of bands
    signal_power *= num_bands
    return 1. - stats.chi2.cdf(
        stats.chi2.ppf(1. - pfa, 2. * n) /
        (1 + signal_power / noise_power), 2. * n)
