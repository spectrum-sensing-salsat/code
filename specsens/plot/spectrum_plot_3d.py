import numpy as np
from scipy import signal
from scipy import fft
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from mpl_toolkits import mplot3d as mpl3d

from specsens import Stft


def spectrum_plot_3d(sig,
                     f_sample,
                     window='flattop',
                     nfft=1024,
                     clip=-60,
                     smooth=.5,
                     crop=None,
                     elev=30,
                     azim=60,
                     type='our',
                     title=None,
                     cmap='viridis',
                     save_png=False):

    if type == 'scipy':
        f, t, ps = signal.spectrogram(sig,
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
        ps = fft.fftshift(ps, axes=0)
        ps = 10.0 * np.log10(ps)

    elif type == 'our':
        sft = Stft(n=nfft, window=window)
        f, t, ps = sft.spectogram(sig, f_sample, normalized=True, dB=True)

    else:
        assert False, 'unknown type'

    if clip is not None:
        ps = clip2d(ps, clip, 0)
    if smooth is not None:
        ps = smooth2d(ps, sigma=smooth)
    if crop is not None:
        ps, f, t = crop2d(ps, f, t, crop)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca(projection='3d')
    # ax.get_proj = lambda: np.dot(mpl3d.axes3d.Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))
    ax.plot_surface(f[:, None],
                    t[None, :],
                    ps,
                    cmap=cmap,
                    rstride=1,
                    cstride=1,
                    alpha=1,
                    antialiased=True)
    ax.view_init(elev=elev, azim=azim)

    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel(r'Time [s]')
    ax.set_zlabel(r'Power [dB]', rotation=90)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    plt.show()

    if save_png:
        plt.savefig('plot3d.png')


def clip2d(ps, lim_min, lim_max):
    (x, y) = np.shape(ps)
    for i in range(x):
        for j in range(y):
            ps[i, j] = max(min(ps[i, j], lim_max), lim_min)
    return ps


def smooth2d(x, sigma):
    return ndimage.filters.gaussian_filter(x, [sigma, sigma], mode='mirror')


def crop2d(ps, f, t, n):
    ps = ps[n:-n, n:-n]
    f = f[n:-n]
    t = t[n:-n]
    return ps, f, t
