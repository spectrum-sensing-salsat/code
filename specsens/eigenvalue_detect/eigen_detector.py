import numpy as np
from scipy import stats
from scipy import linalg

import tqdm

def corr(x, l, prog_bar=False, w_mat=None):
    '''Return correlation matrix'''
    x = np.reshape(x, (-1, 1))
    r = np.zeros((l, l), dtype=np.complex128)
    if prog_bar:
        for i in tqdm.tqdm(range(0, len(x) - l + 1)):
            r += np.dot(x[i:i + l], x[i:i + l].conj().T)
    else:
        for i in range(0, len(x) - l + 1):
            r += np.dot(x[i:i + l], x[i:i + l].conj().T)
    corr_mat = r / (len(x) - l + 1)
    if w_mat is None:
        return corr_mat
    else:
        return np.dot(np.dot(w_mat, corr_mat), w_mat)


def whitening_mat(correlated_noise, l=10, prog_bar=False):
    '''Return whitenning matrix'''
    mat_noise_fil = corr(correlated_noise, l=l, prog_bar=prog_bar)
    return np.linalg.inv(linalg.sqrtm(mat_noise_fil))


# TODO use eigh since matrix is Hermitian

def mme(x, l=10, w_mat=None):
    '''Maximum-minimum eigenvalue'''
    Rx = corr(x, l, prog_bar=False, w_mat=w_mat)
    eigvals = np.abs(linalg.eigvals(Rx))
    return np.max(eigvals) / np.min(eigvals)


def eme(x, l=10, w_mat=None):
    '''Energy with minimum eigenvalue'''
    Rx = corr(x, l, prog_bar=False, w_mat=w_mat)
    eigvals = np.abs(linalg.eigvals(Rx))
    energy = np.sum(np.abs(x)**2.) / len(x)
    return energy / np.min(eigvals)
