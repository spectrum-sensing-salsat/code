import numpy as np
from scipy import stats

# TODO add more asserts on function arguments

def get_pfa(noise_power, thr, n):
    return stats.norm.sf((thr - n * noise_power) / (noise_power * np.sqrt(n)))

def get_pd(noise_power, signal_power, thr, n):
    return stats.norm.sf((thr - n * (noise_power + signal_power)) /
                         (np.sqrt(n) * (noise_power + signal_power)))

def get_thr(noise_power, pfa, n):
    return (np.sqrt(n) * stats.norm.isf(pfa) + n ) * noise_power

def get_roc(noise_power=None, signal_power=None, pfa=None, n=None):
    return stats.norm.sf((noise_power*stats.norm.isf(pfa)-signal_power*np.sqrt(n))/(noise_power + signal_power))

def get_num(noise_power=None, signal_power=None, pfa=None, pd=None):
    return int(np.ceil(((noise_power*stats.norm.isf(pfa)-(noise_power*signal_power*stats.norm.isf(pd)))/signal_power)**2))
