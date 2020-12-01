import numpy as np
from scipy import signal
from scipy import fft
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as mpl3d
# from matplotlib import cm


def clip2d(ps, lim_min, lim_max):
    (x, y) = np.shape(ps)
    for i in range(x):
        for j in range(y):
            ps[i, j] = max(min(ps[i, j], lim_max), lim_min)
    return ps


def smooth2d(x, sigma):
    return ndimage.filters.gaussian_filter(x, [sigma, sigma],
                                                 mode='mirror')


def crop2d(ps, f, t, n):
    ps = ps[n:-n, n:-n]
    f = f[n:-n]
    t = t[n:-n]
    return ps, f, t


def plot3d(sig, f_sample, window='box', nfft=1024, clip=-60, smooth=None, crop=None, elev=30, azim=60):
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
    if clip is not None:
        ps = clip2d(ps, clip, 0)
    if smooth is not None:
        ps = smooth2d(ps, smooth)
    if crop is not None:
        ps, f, t = crop2d(ps, f, t, crop)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca(projection='3d')
    # ax.get_proj = lambda: np.dot(mpl3d.axes3d.Axes3D.get_proj(ax), np.diag([1, 1, 0.5, 1]))
    ax.plot_surface(f[:, None],
                    t[None, :],
                    ps,
                    cmap='viridis',
                    rstride=1,
                    cstride=1,
                    alpha=1,
                    antialiased=True)
    ax.invert_xaxis()
    ax.view_init(elev=elev, azim=azim)
    plt.show()
    # plt.savefig('plot3d.png')
