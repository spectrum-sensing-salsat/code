import numpy as np
from scipy import signal
from scipy import fft

from specsens import util


class Stft:
    def __init__(self, n=1024, window='flattop'):
        assert util.is_power2(n), 'Only power of 2 stft allowed'
        self.n = n
        self.win = signal.get_window(window,
                                     self.n)  # get actual windowing values

    def fft(self, sig, f_sample):
        '''Compute fft and spectrum shifted to natural order.'''
        assert len(sig) == self.n
        dt = 1. / f_sample  # length of each fft in seconds
        f = fft.fftshift(fft.fftfreq(self.n, dt))  # compute frequency indices
        s = fft.fftshift(fft.fft(sig))  # compute actual fft
        return f, s

    def window(self, sig):
        '''Apply the window function.'''
        return sig * self.win

    def psd(self, s, f_sample, normalized=True):
        '''Compute the power spectral density.'''
        ps = np.abs(s)**2.  # power spectral density
        ps /= (self.win**2.).sum()  # normalize with respect to window function
        if normalized:  # normalize to get realistic energy values
            ps /= f_sample
        return ps

    def stft(self, sig, f_sample, normalized=True, dB=True):
        '''Compute the stft on a single segment.'''
        assert len(
            sig) == self.n, 'Signal segment and fft bin length dont match.'
        f, s = self.fft(self.window(sig), f_sample)  # apply windowing
        ps = self.psd(s, f_sample, normalized)
        if dB:
            ps = 10. * np.log10(ps)  # signalpower in dB
        return f, ps

    def spectogram(self, sig, f_sample, normalized=True, dB=True):
        '''Compute spectogram by applying stft on every segment.'''

        # split signal into segments of fft length (discard last 'too short' segement)
        segs = list(
            filter(lambda x: len(x) == self.n,
                   [sig[i:i + self.n]
                    for i in np.arange(0, len(sig), self.n)]))

        # compute psd for every segment and unzip resulting list of tuples
        fl, xl = list(
            zip(*map(lambda x: self.stft(x, f_sample, normalized, dB), segs)))

        # transpose psd matrix to be printable with pcolormesh and
        # to be backwards compatible with scipy functions
        xl = np.asarray(xl).T

        # compute time stamp for every segment
        t = np.asarray(list(map(lambda x: x[0] * self.n * 1. / f_sample, enumerate(segs))))

        # only return first frequency indice list
        f = np.asarray(fl)[0]

        return f, t, xl  # only return first frequency indice list
