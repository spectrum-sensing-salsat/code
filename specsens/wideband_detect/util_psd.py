import numpy as np

def energy(psd):
    '''Returns the signal energy from a power spectral density.'''
    return np.sum(psd)


def power(psd, f):
    '''Returns the signal power from a power spectral density.'''
    df = f[1] - f[0]
    return np.sum(psd) * df

def average(spectogram, start=0, stop=1):
    '''Returns the averaged psd from a spectogram.'''
    return np.mean(spectogram[:, start:stop], axis=1)
