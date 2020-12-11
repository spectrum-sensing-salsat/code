import numpy as np
import matplotlib.pyplot as plt

def psd_energy(psd):
    '''Returns the signal energy from a power spectral density.'''
    return np.sum(psd)


def psd_power(psd, f):
    '''Returns the signal power from a power spectral density.'''
    df = f[1] - f[0]
    return np.sum(psd) * df

def psd_average(spectogram, start=0, stop=1):
    '''Returns the averaged psd from a spectogram.'''
    return np.mean(spectogram[:, start:stop], axis=1)

def plot_matrix(mat, title=None):
    plt.imshow(np.rot90(mat, 1), aspect='auto')
    if title is not None:
        plt.title('Matrix of %s' % title)
    plt.xlabel("Time")
    plt.ylabel("Bands")
    plt.colorbar()
    plt.show()

def store_pic(pic, x, y, n=3, val=1., minimum=0, maximum=1.):
    for i in range(n):
        pic[np.maximum(y - (i+1), minimum), x] = val
        pic[np.minimum(y + (i+1), maximum-1), x] = val
    pic[y, x] = val
    return pic

def scale_matrix(mat, signal_length, band_width, fft_len):
    factor = signal_length*band_width/fft_len
    mat = np.repeat(mat, factor, axis=0)
    return mat, factor

def get_pfa(result, truth):
    truth = (truth == -100) * np.full_like(truth, 1.)
    result = truth * result
    return np.sum(result, axis=(0,1)) / np.sum(truth, axis=(0,1))

def get_pd(result, truth):
    truth = (truth != -100) * np.full_like(truth, 1.)
    result = truth * result
    return np.sum(result, axis=(0,1)) / np.sum(truth, axis=(0,1))
