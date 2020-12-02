import numpy as np


def round_power2(x):
    '''Round up to nearest power of two.'''
    assert x > 0., 'x must be greater than 0'
    assert not np.iscomplex(x), 'x cannot be complex'
    return int(np.power(2., np.ceil(np.log2(x))))


def is_power2(n):
    '''Check least significant bit for power of two.'''
    assert n > 0., 'n must be greater than 0'
    return (n != 0) and (n & (n - 1) == 0)


def sample_time(f_sample, num_samples):
    ''' Retruns the total signal time in seconds.'''
    assert f_sample > 0., 'f_sample must be greater than 0'
    assert num_samples > 0, 'num_samples must be greater than 0'
    return num_samples / f_sample


def get_signal_length(f_sample, t_sec):
    ''' Retruns the signals length in number of samples.'''
    assert f_sample > 0., 'f_sample must be greater than 0'
    assert t_sec > 0., 't_sec must be greater than 0'
    return f_sample * t_sec


def dB_power(x):
    '''Compute dB for power signals.'''
    assert x > 0., 'x must be greater than 0'
    assert not np.iscomplex(x), 'x cannot be complex'
    return 10. * np.log10(x)


def dB(x):
    '''Compute dB for amplitude signals.'''
    assert x > 0., 'x must be greater than 0'
    assert not np.iscomplex(x), 'x cannot be complex'
    return 20. * np.log10(x)


def dB_to_factor_power(x):
    '''Compute factor for power signals.'''
    assert not np.iscomplex(x), 'x cannot be complex'
    return 10.**(x / 10.)


def dB_to_factor(x):
    '''Compute factor for amplitude signals.'''
    assert not np.iscomplex(x), 'x cannot be complex'
    return 10.**(x / 20.)


def signal_power(x, dB=True):
    '''Returns the signal power.'''
    assert x.ndim == 1, 'x must have exactly 1 dimension'
    p = np.sum(np.abs(x)**2.) / len(x)
    if dB:
        p = dB_power(p)
    return p


def snr(x, w, dB=True):
    '''Returns the signal to noise ratio between x and w.'''
    assert x.ndim == 1, 'x must have exactly 1 dimension'
    assert w.ndim == 1, 'y must have exactly 1 dimension'
    snr = signal_power(x, dB=False) / signal_power(w, dB=False)
    if dB:
        snr = dB_power(snr)
    return snr


def signal_energy(x, t_sec):
    '''Returns the signal energy.'''
    assert x.ndim == 1, 'x must have exactly 1 dimension'
    assert t_sec > 0., 't_sec must be greater than 0'
    return signal_power(x, dB=False) * t_sec


def get_energy_from_psd(psd):
    '''Returns the signal energy from a power spectral density.'''
    return np.sum(psd)


def get_power_from_psd(psd, f):
    '''Returns the signal power from a power spectral density.'''
    df = f[1] - f[0]
    return np.sum(psd) * df
