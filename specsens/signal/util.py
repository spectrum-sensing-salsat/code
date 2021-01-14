import numpy as np
import math
from scipy import fft

def round_power2(x):
    '''Round up to nearest power of two'''
    assert np.all(x > 0.), 'x must be greater than 0'
    assert np.all(np.isreal(x)), 'x cant be complex'
    return int(np.power(2., np.ceil(np.log2(x))))


def is_power2(n):
    '''Check least significant bit for power of two'''
    assert n > 0., 'n must be greater than 0'
    return (n != 0) and (n & (n - 1) == 0)


def sample_time(f_sample, num_samples):
    '''Returns the total signal time in seconds'''
    assert f_sample > 0., 'f_sample must be greater than 0'
    assert num_samples > 0, 'num_samples must be greater than 0'
    return num_samples / f_sample


def signal_length(f_sample, t_sec):
    '''Returns the signal length in number of samples'''
    assert f_sample > 0., 'f_sample must be greater than 0'
    assert t_sec > 0., 't_sec must be greater than 0'
    return int(f_sample * t_sec)


def dB_power(x):
    '''Compute dB for power signals'''
    assert np.all(x > 0.), 'x must be greater than 0'
    assert np.all(np.isreal(x)), 'x cant be complex'
    return 10. * np.log10(x)


def dB(x):
    '''Compute dB for amplitude signals'''
    assert np.all(x > 0.), 'x must be greater than 0'
    assert np.all(np.isreal(x)), 'x cant be complex'
    return 20. * np.log10(x)


def dB_to_factor_power(x):
    '''Compute factor for power signals'''
    assert np.all(np.isreal(x)), 'x cant be complex'
    return 10.**(x / 10.)


def dB_to_factor(x):
    '''Compute factor for amplitude signals'''
    assert np.all(np.isreal(x)), 'x cant be complex'
    return 10.**(x / 20.)


def signal_power(x, dB=True):
    '''Returns the signal power'''
    assert x.ndim == 1, 'x must have exactly 1 dimension'
    p = np.sum(np.abs(x)**2.) / len(x)
    if dB:
        p = dB_power(p)
    return p


def snr(x, w, dB=True):
    '''Returns the signal to noise ratio between x and w'''
    assert x.ndim == 1, 'x must have exactly 1 dimension'
    assert w.ndim == 1, 'w must have exactly 1 dimension'
    snr = signal_power(x, dB=False) / signal_power(w, dB=False)
    if dB:
        snr = dB_power(snr)
    return snr


def signal_energy(x, t_sec):
    '''Returns the signal energy'''
    assert x.ndim == 1, 'x must have exactly 1 dimension'
    assert t_sec > 0., 't_sec must be greater than 0'
    return signal_power(x, dB=False) * t_sec


def check_parseval(x, length, f_sample, print_eng=False):
    '''
    Checks that parsevals theorem is not violated.
    This is a good sanity check. If parsevals theorem is violated something
    has gone really bad or is broken.
    '''
    time_eng = np.sum(np.abs(x)**2 * length)
    freq_eng = np.sum(np.abs(fft.fft(x))**2) / f_sample
    if print_eng:
        print("\nTime energy:  %.2f" % (time_eng))
        print("Freq energy:  %.2f" % (freq_eng))
    assert math.isclose(time_eng, freq_eng), 'parsevals theorem violated'

def dB_rel_err(act, val, dB=True):
    if dB:
        act = dB_to_factor_power(act)
        val = dB_to_factor_power(val)
    return dB_power(np.abs((act - val) / val))
