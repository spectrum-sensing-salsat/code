import numpy as np
from scipy import stats

# TODO add more asserts on function arguments

def get_thr(noise_power, pfa, n):
    return stats.chi2.ppf(1.-pfa, 2.*n)*noise_power/2.


def get_pfa(noise_power, thr, n):
    return 1. - stats.chi2.cdf(2.*thr/noise_power, 2.*n)


def get_pd(noise_power, signal_power, thr, n):
    return 1. - stats.chi2.cdf(2.*thr/(noise_power+signal_power), 2.*n)


def get_roc(noise_power=None, signal_power=None, snr=None, pfa=None, n=None):
    if snr is not None:
        snr = snr
    elif noise_power is not None and signal_power is not None:
        snr = signal_power/noise_power
    else:
        assert False, 'either supply snr or signal and noise power'
    return 1. - stats.chi2.cdf(stats.chi2.ppf(1.-pfa, 2.*n)/(1+signal_power/noise_power), 2.*n)
