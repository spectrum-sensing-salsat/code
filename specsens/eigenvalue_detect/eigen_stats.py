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
