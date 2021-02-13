import numpy as np
from scipy import stats
from scipy import linalg
import TracyWidom

from specsens import util


def mme_thr(Ns, L, Pfa=0.1, M=1):
    tw = TracyWidom.TracyWidom(beta=1)
    fac1 = (Ns**.5 + (M * L)**.5)**2 / (Ns**.5 - (M * L)**.5)**2
    fac2 = 1 + ((Ns**.5 + (M * L)**.5)**(-2 / 3) /
                (Ns * M * L)**(1 / 6)) * tw.cdfinv(1 - Pfa)
    return fac1 * fac2


def mme_pfa(Ns, L, thr, M=1):
    tw = TracyWidom.TracyWidom(beta=1)
    mu = ((Ns - 1)**.5 + (M * L)**.5)**2
    v = ((Ns - 1)**.5 + (M * L)**.5) * (1 / ((Ns - 1)**.5) + 1 /
                                        ((M * L)**.5))**(1 / 3)
    fac = (thr * (Ns**.5 - (M * L)**.5)**2. - mu) / v
    return 1 - tw.cdf(fac)


def mme_pd(Ns, L, thr, noise_power, sig_eigv_min, sig_eigv_max, M=1, dB=True):
    if dB:
        noise_power = util.dB_to_factor_power(noise_power)
    tw = TracyWidom.TracyWidom(beta=1)
    mu = ((Ns - 1)**.5 + (M * L)**.5)**2
    v = ((Ns - 1)**.5 + (M * L)**.5) * (1 / ((Ns - 1)**.5) + 1 /
                                        ((M * L)**.5))**(1 / 3)
    fac = (thr * Ns +
           (Ns * (thr * sig_eigv_min - sig_eigv_max)) / noise_power - mu) / v
    return 1 - tw.cdf(fac)


def eme_thr(Ns, L, Pfa=0.1, M=1):
    fac1 = (2. / (M * Ns))**.5 * stats.norm.isf(Pfa) + 1
    fac2 = Ns / (Ns**.5 - (M * L)**.5)**2
    return fac1 * fac2


def eme_pfa():
    pass


def eme_pd():
    pass
