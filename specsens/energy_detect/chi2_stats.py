import numpy as np
from scipy import stats

# TODO add more asserts on function arguments

from specsens import util

def get_pfa(noise_power, thr, n, dB=True):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
    return 1. - stats.chi2.cdf(2. * thr / noise_power, 2. * n)


def get_pd(noise_power, signal_power, thr, n, dB=True, bands=1):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
        signal_power = util.dB_to_factor_power(signal_power)
    signal_power *= bands
    return 1. - stats.chi2.cdf(2. * thr / (noise_power + signal_power), 2. * n)


def get_thr(noise_power, pfa, n, dB=True):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
    return stats.chi2.ppf(1. - pfa, 2. * n) * noise_power / 2.


def get_roc(noise_power=None, signal_power=None, pfa=None, n=None, dB=True, bands=1):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
        signal_power = util.dB_to_factor_power(signal_power)
    signal_power *= bands
    return 1. - stats.chi2.cdf(stats.chi2.ppf(1. - pfa, 2. * n) / (1 + signal_power / noise_power), 2. * n)
