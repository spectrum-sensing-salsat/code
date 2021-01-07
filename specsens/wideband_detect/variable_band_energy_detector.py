import numpy as np
from scipy import signal
from scipy import fft

from specsens import edge_detector
from specsens import chi2_stats


class VariableBandEnergyDetector():
    def __init__(self, f_sample=1e6, fft_len=1024, freqs=None, noise_power=0., pfa=0.1, smooth=10., scale=5, min_height=0.2, min_freq=3e4):
        self.f_sample = f_sample  # sample rate used with the
        self.fft_len = fft_len  # length of the fft used to compute the power spectrum
        self.freqs = freqs  # frequency band, x axis, used by the power spectrum
        self.noise_power = noise_power  # power of noise in dB
        self.pfa = pfa  # probability of false alarm
        self.ps_list = np.zeros(shape=(fft_len, 0))  # list to smooth power spectrum
        self.min_freq = min_freq  # minimum distance between to edges

    def ps_smooth(self, ps):
        if np.size(self.ps_list, 1) >= 10:
            self.ps_list = np.delete(self.ps_list, 0, axis=1)
        self.ps_list = np.concatenate((self.ps_list, np.array([ps]).T), axis=1)
        return np.mean(self.ps_list, axis=1)

    def threshold(self, v):
        thr = chi2_stats.get_thr(noise_power=self.noise_power,
                                 pfa=self.pfa,
                                 n=v[3],
                                 dB=True)
        return v[0] > thr

    def detect(self, ps):
        ps = self.ps_smooth(ps)
        ps_dB = 10. * np.log10(ps)
        prod, peak, peakf = edge_detector(ps_dB,
                                        self.freqs,
                                        scale=5,
                                        min_height=0.2,
                                        min_freq=self.min_freq)
        dics = self.energies(peak, ps)
        dics = {k: v for k, v in dics.items() if self.threshold(v)}
        return ps, peak, peakf, dics

    def energies(self, peak, ps):
        energ = {}
        if len(peak) == 0:
            return energ
        energ[self.cen(0, peak[0])] = (self.en(ps, 0, peak[0]),
                                       0, peak[0], np.abs(peak[0] - 0))
        for i, _ in enumerate(peak[:-1]):
            energ[self.cen(peak[i], peak[i + 1])] = (self.en(ps, peak[i],
                                                             peak[i + 1]), peak[i], peak[i + 1], np.abs(peak[i + 1] - peak[i]))
        energ[self.cen(peak[-1], len(ps))] = (self.en(ps, peak[-1],
                                                      len(ps)), peak[-1], len(ps), np.abs(peak[-1] - len(ps)))
        return energ

    def en(self, ps, start, stop):
        return np.sum(ps[start:stop]) / np.abs(stop - start)

    def cen(self, start, stop):
        return start + np.abs(stop - start) / 2.
