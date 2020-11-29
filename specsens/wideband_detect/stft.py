import numpy as np
from scipy import signal
from scipy import fft

from specsens import util

class Stft:

    def __init__(self, n=1024, window='flattop'):
        assert ss.util.is_power2(n)
        self.n = n
        self.window = signal.get_window(window, self.n)

    def fft(self, sig, f_sample):
        '''Compute FFT and spectrum shifted to natural order.'''
        dt = 1./f_sample
        f = fft.fftshift(fft.fftfreq(len(sig), dt))
        s = fft.fftshift(fft.fft(sig))
        return f, s

    def apply_window(self, sig):
        return sig * self.window

    def stft_helper(self, sig, f_sample):
        return self.fft(self.apply_window(sig), f_sample)

    def stft(self, sig, f_sample):
        segs = [sig[i:i + self.n] for i in range(0, len(sig), self.n)]
        for seg in segs:
            if len(seg) != self.n:
                break
            f, x = self.stft_helper(seg, f_sample)
            yield f, x

    def apply_psd(self, sig, f_sample):
#         return np.abs(sig)**2. / (len(sig) * f_sample)
        return np.abs(sig)**2. / (f_sample * (self.window**2).sum())

    def psd(self, sig, f_sample):
        for f, x in self.stft(sig, f_sample):
            yield f, self.apply_psd(x, f_sample)

    def spectogram(self, sig, f_sample, dB=True):
        tl = list()
        xl = list()
        if dB:
            dB_func = lambda x : ss.util.dB(x)
        else:
            dB_func = lambda x : x
        for i, (f, x) in enumerate(self.psd(sig, f_sample)):
            tl.append(i*self.n*1./f_sample)
            xl.append(x)
        xl = np.swapaxes(xl, 0, 1)
        return f, tl, xl
