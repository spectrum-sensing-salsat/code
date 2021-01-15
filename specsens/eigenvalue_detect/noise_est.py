import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from scipy import stats
from scipy import linalg

from specsens import util
from specsens import eigen_detector


def mdl_citerion(L, N, eigs, M):
    '''Minimum description length criterion.'''
    eigs = eigs[M:]  # select the last M eigenvalues
    prod = np.prod(eigs**(1 / (L - M)))
    summ = np.sum(eigs) / (L - M)
    return -(L - M) * N * np.log(
        prod / summ) + .5 * M * (2 * L - M) * np.log(N)


def marchenko_pastur_pdf_list(var, c, res):
    '''Marchenko pastur pdf returning a list.'''
    v_max, v_min = var * (1 + c**.5)**2., var * (1 - c**.5)**2.
    v = np.linspace(v_min, v_max, res)
    return v, ((v - v_min) * (v_max - v))**.5 / (2. * np.pi * var * v * c)


def marchenko_pastur_pdf(var, c, v):
    '''Marchenko pastur pdf returning a scalar.'''
    v_max, v_min = var * (1 + c**.5)**2., var * (1 - c**.5)**2.
    if v > v_max or v < v_min:
        return 0
    else:
        return ((v - v_min) * (v_max - v))**.5 / (2. * np.pi * var * v * c)


def kde_dist(eigs, v, weights=False):
    '''Empirical distribution using kernel density estimation.'''
    if weights:
        w = np.linspace(1., 1.2, len(eigs))
        return stats.gaussian_kde(eigs, bw_method='scott',
                                  weights=w).evaluate(v)
    else:
        return stats.gaussian_kde(eigs, bw_method='scott').evaluate(v)


def goodness_fit_kde(c, beta, res, eigs, var):
    '''Goodness of fit using kde.'''
    v, mp = marchenko_pastur_pdf_list(var, (1 - beta) * c, res=res)
    kde = kde_dist(eigs, v)
    return linalg.norm(kde - mp)


def hist_dist(eigs, l):
    '''Empirical distribution using histogram.'''
    ys, xs = np.histogram(eigs, bins=l // 2, density=True)
    xs = (xs + np.roll(xs, -1))[:-1] / 2.0
    return xs, ys


def goodness_fit_hist(c, beta, l, eigs, var):
    '''Goodness of fit using histogram.'''
    xs, ys = hist_dist(eigs, l)
    mp = list(map(lambda x: marchenko_pastur_pdf(var, (1 - beta) * c, x), xs))
    return linalg.norm(ys - mp)


def mle(c, beta, eigs, var):
    '''Maximum likelihood estimation.'''
    return np.sum(
        list(map(lambda x: marchenko_pastur_pdf(var, (1 - beta) * c, x),
                 eigs)))


def estimate(x, n, l=50, res=1000, dB=True, prints=False, true_power=None):
    '''Estimate noise power directly from covariance eigenvalues.'''
    assert len(x) == n, 'Length does not match n'
    assert len(x) > l, 'Length cant be smaller than l'

    # claculate covariance matrix
    mat = eigen_detector.corr(x, l)

    if prints:
        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(mat))
        plt.colorbar()
        plt.title('Matrix eigenvalues')
        plt.show()

    # get sorted eigenvalues (descending)
    eigs = np.sort(np.abs(linalg.eigvals(mat)))[::-1]

    if prints:
        plt.figure(figsize=(8, 6))
        plt.plot(eigs, 'o')
        plt.grid(linewidth=0.5)
        plt.xlabel(r'Eigenvalue number')
        plt.ylabel(r'Magnitude')
        plt.title('Sorted eigenvalues')
        plt.show()

    # calculate mdl
    mdl = list(map(lambda x: mdl_citerion(l, n, eigs, x), np.arange(l)))

    if prints:
        plt.figure(figsize=(8, 6))
        plt.plot(mdl, 'o')
        plt.grid(linewidth=0.5)
        plt.yscale('log')
        plt.xlabel(r'Eigenvalues')
        plt.ylabel(r'Information needed')
        plt.title('MDL function')
        plt.show()

    # get optimal m (find first noise eignvalue index)
    m = np.argmin(mdl)

    # noise only eigenvalues
    noise_eigs = eigs[m:]

    # estimate noise from mean noise eigenvalues
    noise_est_mean = np.mean(noise_eigs)

    ### use model fitting for more accurate estimation ###

    # calculate parameters needed for distribution
    c = l / n
    beta = m / l

    # calculate endpoints to use for model fitting
    pis = np.linspace(noise_eigs[-1] / (1 - c**.5)**2 * .9,
                      noise_eigs[0] / (1 + c**.5)**2 * 1.1, res)

    # calculate differences for all pis using kde
    diffs_kde = list(
        map(lambda x: goodness_fit_kde(c, beta, res, noise_eigs, var=x), pis))
    # and select optimal one
    noise_est_fit_kde = pis[np.argmin(diffs_kde)]

    # compute likelihoods
    likelihoods = list(map(lambda x: mle(c, beta, noise_eigs, x), pis))
    # and select maximum likelihood
    noise_est_mle = pis[np.argmax(likelihoods)]

    # calculate differences for all pis using kde
    diffs_hist = list(
        map(lambda x: goodness_fit_hist(c, beta, l, noise_eigs, var=x), pis))
    # and select optimal one
    noise_est_fit_hist = pis[np.argmin(diffs_hist)]

    if prints:
        v_truth, mp_truth = marchenko_pastur_pdf_list(
            util.dB_to_factor_power(true_power), c, res)
        v_mean, mp_mean = marchenko_pastur_pdf_list(noise_est_mean, c, res)
        v_hist, mp_hist = marchenko_pastur_pdf_list(noise_est_fit_hist, c, res)
        v_kde, mp_kde = marchenko_pastur_pdf_list(noise_est_fit_kde, c, res)
        v_mle, mp_mle = marchenko_pastur_pdf_list(noise_est_mle, c, res)
        plt.figure(figsize=(8, 6))
        plt.hist(noise_eigs, bins=l // 2, density=True, alpha=0.5,
                 aa=True, label='Eigenvalue distribution')
        plt.plot(v_truth, mp_truth, 'g-', aa=True, label='Actual')
        plt.plot(v_mean, mp_mean, 'r-', aa=True, label='Mean')
        plt.plot(v_hist, mp_hist, 'y-', aa=True, label='Hist')
        plt.plot(v_kde, mp_kde, 'b-', aa=True, label='KDE')
        plt.plot(v_kde, kde_dist(noise_eigs, v_kde), aa=True, label='KDE dist')
        plt.plot(v_mle, mp_mle, 'k-', aa=True, label='MLE')
        plt.grid(linewidth=0.5)
        plt.legend(loc=0)
        plt.show()

    # return results in dB if requested
    if dB:
        noise_est_mean = util.dB_power(noise_est_mean)
        noise_est_fit_hist = util.dB_power(noise_est_fit_hist)
        noise_est_fit_kde = util.dB_power(noise_est_fit_kde)
        noise_est_mle = util.dB_power(noise_est_mle)

    return noise_est_mean, noise_est_fit_hist, noise_est_fit_kde, noise_est_mle


def estimate_quick(x, n, l=50, res=1000, dB=True):
    '''Estimate noise power directly from covariance eigenvalues.'''
    assert len(x) == n, 'Length does not match n'
    assert len(x) > l, 'Length cant be smaller than l'

    # claculate covariance matrix
    mat = eigen_detector.corr(x, l)

    # get sorted eigenvalues (descending)
    eigs = np.sort(np.abs(linalg.eigvals(mat)))[::-1]

    # calculate mdl
    mdl = list(map(lambda x: mdl_citerion(l, n, eigs, x), np.arange(l)))

    # get optimal m (find first noise eignvalue index)
    m = np.argmin(mdl)

    # noise only eigenvalues
    noise_eigs = eigs[m:]

    # estimate noise from mean noise eigenvalues
    noise_est_mean = np.mean(noise_eigs)

    # return result in dB if requested
    if dB:
        noise_est_mean = util.dB_power(noise_est_mean)

    return noise_est_mean
