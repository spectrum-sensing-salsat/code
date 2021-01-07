import numpy as np
from scipy import stats
from scipy import linalg

from specsens import TracyWidom


def mme_thr(Ns, L, Pfa=0.1, M=1):
    tw = TracyWidom(beta=1)
    fac1 = (np.sqrt(Ns) + np.sqrt(M * L))**2 / (np.sqrt(Ns) -
                                                np.sqrt(M * L))**2
    fac2 = 1 + ((np.sqrt(Ns) + np.sqrt(M * L))**(-2 / 3) /
                (Ns * M * L)) * tw.cdfinv(1 - Pfa)
    return fac1 * fac2


def eme_thr(Ns, L, Pfa=0.1, M=1):
    fac1 = np.sqrt(2. / (M * Ns)) * stats.norm.isf(Pfa) + 1
    fac2 = Ns / (np.sqrt(Ns) - np.sqrt(M * L))**2
    return fac1 * fac2


def corr(x, l):
    x = np.reshape(x, (-1, 1))
    r = np.zeros((l, l), dtype=np.complex128)
    for i in range(0, len(x) - l):
        r += np.dot(x[i:i + l], x[i:i + l].conj().T)
    return r / (len(x) - l)

# TODO use eigh since matrix is Hermitian

def mme(x, l=10):
    Rx = corr(x, l)
    eigvals = np.abs(linalg.eigvals(Rx))
    return np.max(eigvals) / np.min(eigvals)


def eme(x, l=10):
    Rx = corr(x, l)
    eigvals = np.abs(linalg.eigvals(Rx))
    energy = np.sum(np.abs(x)**2.) / len(x)
    return energy / np.min(eigvals)
