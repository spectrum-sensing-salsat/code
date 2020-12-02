import numpy as np
from scipy import signal
from scipy import fft

from specsens import util
from specsens import Stft


class FreqEnergyDetector():

    def __init__(self, num_bands=8, f_sample=1e6, fft_len=1024, freqs=None):
        self.num_bands = num_bands  # number of bands
        self.f_sample = f_sample  # sample rate used with the
        self.fft_len = fft_len  # length of the fft used to compute the power spectrum
        self.freqs = freqs  # frequency band, x axis, used by the power spectrum
        self.di = fft_len / num_bands  # calculate the points per band
        self.df = freqs[1] - freqs[0]  # calculate the frequency width per band
        assert np.isclose(self.df, (self.f_sample / self.fft_len))

    def segs(self, ps):
        f_segs = list()
        ps_segs = list()
        for i in range(int(self.num_bands)):
            idx1 = int(i * self.di)
            idx2 = int((i + 1) * self.di)
            f_segs.append(self.freqs[idx1:idx2])
            ps_segs.append(ps[idx1:idx2])
        return f_segs, ps_segs

    def detect(self, ps):
        f_segs, ps_segs = self.segs(ps)
        return list(map(lambda x: np.sum(x), ps_segs))
