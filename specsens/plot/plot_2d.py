import numpy as np
from scipy import signal
from scipy import fft
import matplotlib as mpl
import matplotlib.pyplot as plt
from varname import nameof

from specsens import util
from specsens import Stft

def plot2d(sig, f_sample, window='flattop', nfft=1024, dB=True, type='our', title=None):
    if type == 'matplotlib':
        plt.figure(figsize=(8, 6))
        if title is not None:
            plt.title('PSD of %s' % title)
        plt.specgram(sig,
                     NFFT=nfft,
                     Fs=f_sample,
                     scale='dB' if dB else 'linear',
                     cmap='viridis',
                     noverlap=0,
                     mode='psd',
                     window=signal.get_window(window, nfft))
    elif type == 'scipy':
        plt.figure(figsize=(8, 6))
        if title is not None:
            plt.title('PSD of %s' % title)
        f, t, x = signal.spectrogram(both,
                             sample_freq,
                             return_onesided=False,
                             window='flattop',
                             nperseg=nfft,
                             nfft=nfft,
                             noverlap=0,
                             detrend=False,
                             scaling='density',
                             mode='psd')
        f = fft.fftshift(f)
        x = fft.fftshift(x, axes=0)
        if dB:
            x = util.dB_power(x)
        plt.pcolormesh(t, f, x, shading='virdis')
    elif type == 'our':
        plt.figure(figsize=(8, 6))
        if title is not None:
            plt.title('PSD of %s' % title)
        sft = Stft(n=nfft, window='flattop')
        f, t, x = sft.spectogram(sig, f_sample, normalized=True, dB=dB)
        plt.pcolormesh(t,
                       f,
                       x,
                       shading='flat',
                       cmap='viridis',
                       snap=True,
                       vmin=np.min(x),
                       vmax=np.max(x))
    else:
        assert False, 'unknown type'

    plt.xlabel('Time')
    plt.ylabel('Frequency [dB]' if dB else 'Frequency [linear]')
    plt.show()
