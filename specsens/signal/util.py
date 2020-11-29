import numpy as np

def round_power2(x):
    assert x > 0., 'x must be greater than 0'
    assert not np.iscomplex(x), 'x cannot be complex'
    return int(np.power(2., np.ceil(np.log2(x))))

def is_power2(n):
    '''Check least significant bit for power of two.'''
    assert n > 0., 'n must be greater than 0'
    return (n != 0) and (n & (n-1) == 0)

def sample_time(f_sample, num_samples):
    assert f_sample > 0., 'f_sample must be greater than 0'
    assert num_samples > 0, 'num_samples must be greater than 0'
    return num_samples / f_sample

def get_signal_length(f_sample, t_sec):
    assert f_sample > 0., 'f_sample must be greater than 0'
    assert t_sec > 0., 't_sec must be greater than 0'
    return f_sample * t_sec

def dB_power(x):
    assert x > 0., 'x must be greater than 0'
    assert not np.iscomplex(x), 'x cannot be complex'
    return 10. * np.log10(x)

def dB(x):
    assert x > 0., 'x must be greater than 0'
    assert not np.iscomplex(x), 'x cannot be complex'
    return 20. * np.log10(x)

def dB_to_factor_power(x):
    assert not np.iscomplex(x), 'x cannot be complex'
    return 10.**(x / 10.)

def dB_to_factor(x):
    assert not np.iscomplex(x), 'x cannot be complex'
    return 10.**(x / 20.)

def signal_power(x, dB=True):
    assert x.ndim == 1, 'x must have exactly 1 dimension'
    p = np.sum(np.abs(x)**2.) / len(x)
    if dB:
        return dB_power(p)
    else:
        return p

def snr(x, y, dB=True):
    assert x.ndim == 1, 'x must have exactly 1 dimension'
    assert y.ndim == 1, 'y must have exactly 1 dimension'
    snr = signal_power(x, dB=False) / signal_power(y, dB=False)
    if dB:
        return dB_power(snr)
    else:
        return snr

def signal_energy(x, t_sec):
    assert x.ndim == 1, 'x must have exactly 1 dimension'
    assert t_sec > 0., 't_sec must be greater than 0'
    return signal_power(x, dB=False) * t_sec
