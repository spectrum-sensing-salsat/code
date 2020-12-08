import numpy as np
from scipy import signal
from scipy import fft

from specsens import util


class Stft:
    def __init__(self, n=1024, window='flattop'):
        assert util.is_power2(n), 'Only power of 2 stft allowed'
        self.n = n
        self.win = signal.get_window(window, self.n)

    def fft(self, sig, f_sample):
        '''Compute fft and spectrum shifted to natural order.'''
        assert len(sig) == self.n
        dt = 1. / f_sample
        f = fft.fftshift(fft.fftfreq(self.n, dt))
        s = fft.fftshift(fft.fft(sig))
        return f, s

    def window(self, sig):
        '''Apply the window function.'''
        return sig * self.win

    def psd(self, s, f_sample, normalized=True):
        '''Compute the power spectral density.'''
        ps = np.abs(s)**2. / (self.win**2.).sum()
        if normalized:
            ps /= f_sample
        return ps

    def stft(self, sig, f_sample, normalized=True, dB=True):
        '''Compute the stft on a single segment.'''
        assert len(sig) == self.n, 'Signal segment and fft bin length dont match.'
        f, s = self.fft(self.window(sig), f_sample)
        ps = self.psd(s, f_sample, normalized)
        if dB:
            ps = 10. * np.log10(ps)  # signalpower in dB
        return f, ps

    def spectogram(self, sig, f_sample, normalized=True, dB=True):
        '''Compute spectogram by applying stft on every segment.'''
        tl = []
        xl = []
        segs = filter(lambda x: len(x) == self.n,
                      [sig[i:i + self.n] for i in range(0, len(sig), self.n)])
        for i, seg in enumerate(segs):
            f, x = self.stft(seg, f_sample, normalized, dB)
            tl.append(i * self.n * 1. / f_sample)
            xl.append(x)
        xl = np.swapaxes(xl, 0, 1)
        return f, np.array(tl), xl
