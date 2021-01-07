import numpy as np
from scipy import signal
from scipy import fft
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

from specsens import util
from specsens import Stft


def spectrum_plot_2d(sig,
                     f_sample,
                     window='flattop',
                     nfft=1024,
                     dB=True,
                     type='our',
                     title=None,
                     cmap='viridis',
                     grid=False):

    if type == 'matplotlib':
        plt.figure(figsize=(8, 6))
        if title is not None:
            plt.title('Spectogram of %s' % title)
        plt.specgram(sig,
                     NFFT=nfft,
                     Fs=f_sample,
                     scale='dB' if dB else 'linear',
                     cmap=cmap,
                     noverlap=0,
                     mode='psd',
                     window=signal.get_window(window, nfft))

    elif type == 'scipy':
        plt.figure(figsize=(8, 6))
        if title is not None:
            plt.title('Spectogram of %s' % title)
        f, t, x = signal.spectrogram(sig,
                                     f_sample,
                                     return_onesided=False,
                                     window=window,
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
        plt.pcolormesh(t,
                       f,
                       x,
                       cmap=cmap,
                       vmin=np.min(x),
                       vmax=np.max(x))
        if grid:
            plt.grid(True, color='C0')

    elif type == 'our':
        plt.figure(figsize=(8, 6))
        if title is not None:
            plt.title('Spectogram of %s' % title)
        sft = Stft(n=nfft, window=window)
        f, t, x = sft.spectogram(sig, f_sample, normalized=True, dB=dB)
        plt.pcolormesh(t,
                       f,
                       x,
                       cmap=cmap,
                       vmin=np.min(x),
                       vmax=np.max(x))
        if grid:
            plt.grid(True, color='C0')

    else:
        assert False, 'unknown type'

    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.gca().invert_yaxis()
    plt.show()
