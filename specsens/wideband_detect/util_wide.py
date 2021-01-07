import numpy as np
import math
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

def plot_matrix(mat, title=None, xlabel='Time', ylabel='Band', plabel=None, binary=False):
    '''Plot a matrix.'''
    plt.imshow(mat, aspect='auto')
    if title is not None:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cbar = plt.colorbar()
    if plabel is not None:
        cbar.set_label(plabel)
    if binary is True:
        cbar.set_ticks([0, 1])
    plt.show()

def store_pic(pic, x, y, n=3, val=1., minimum=0, maximum=1.):
    assert False
    for i in range(n):
        pic[np.maximum(y - (i+1), minimum), x] = val
        pic[np.minimum(y + (i+1), maximum-1), x] = val
    pic[y, x] = val
    return pic

def scale_matrices(mat1, mat2, axis=1):
    '''Scale matrices s.t. they can be compared.'''

    # get shapes we are supposed to scale
    shapes = (np.shape(mat1)[axis], np.shape(mat2)[axis])

    # find least common multiple
    lcm = abs(shapes[0] * shapes[1]) // math.gcd(shapes[0], shapes[1])

    # scale matrices accordingly
    mat1 = np.repeat(mat1, lcm // shapes[0], axis=axis)
    mat2 = np.repeat(mat2, lcm // shapes[1], axis=axis)

    return mat1, mat2


def mat_pfa(result, truth):
    '''Compute probability of false alarm from matrices.'''
    result, truth = scale_matrices(result, truth)
    truth = (truth == -100) * np.full_like(truth, 1.)
    result = truth * result
    return np.sum(result, axis=(0, 1)) / np.sum(truth, axis=(0, 1))


def mat_pd(result, truth):
    '''Compute probability of detection from matrices.'''
    result, truth = scale_matrices(result, truth)
    truth = (truth != -100) * np.full_like(truth, 1.)
    result = truth * result
    return np.sum(result, axis=(0, 1)) / np.sum(truth, axis=(0, 1))
