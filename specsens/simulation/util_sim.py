import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import time


from specsens import util


# !!! deprecated !!!
# def runtime_stats(current_time, total_itr, current_itr):
#     if current_time is None:  # first iteration cant predict time
#         current_time = time.time()
#         return float('inf'), 0.0, current_time
#     delta_time = time.time() - current_time
#     current_time = time.time()
#     remaining_itr = total_itr - current_itr
#     remaining_time = delta_time * remaining_itr
#     percent_done = current_itr / total_itr * 100.0
#     return remaining_time, percent_done, current_time


def print_convergence(gens, pfas, pds, theo_pfa, theo_pd):
    plt.figure(figsize=(8, 6))
    for i in range(1, gens):
        inter = np.mean(pfas[0:i])
        plt.plot(i, inter, 'o', c='C0', markersize=2, aa=True)
    sim = plt.axhline(np.mean(pfas), c='C1', ls='--', aa=True)
    theo = plt.axhline(theo_pfa, c='C2', ls='-', aa=True)
    plt.legend(handles=(sim, theo), labels=('Simulation', 'Theory'), loc=0)
    plt.xlabel(r'Generations')
    plt.ylabel(r'$P_{fa}$ (Probability of false alarm)')
    plt.grid(linewidth=0.5)
    plt.show()

    plt.figure(figsize=(8, 6))
    for i in range(1, gens):
        inter = np.mean(pds[0:i])
        plt.plot(i, inter, 'o', c='C0', markersize=2, aa=True)
    sim = plt.axhline(np.mean(pds), c='C1', ls='--', aa=True)
    theo = plt.axhline(theo_pd, c='C2', ls='-', aa=True)
    plt.legend(handles=(sim, theo), labels=('Simulation', 'Theory'), loc=0)
    plt.xlabel(r'Generations')
    plt.ylabel(r'$P_d$ (Probability of detection)')
    plt.grid(linewidth=0.5)
    plt.show()


def print_distribution(eng_both,
                       eng_noise,
                       n,
                       signal_power,
                       noise_power,
                       threshold,
                       bins=100):

    plt.figure(figsize=(8, 6))

    # histograms from simulations
    plt.hist(eng_both,
             bins,
             density=True,
             color='C0',
             alpha=0.6,
             aa=True,
             label='Signal present')
    plt.hist(eng_noise,
             bins,
             density=True,
             color='C1',
             alpha=0.6,
             aa=True,
             label='Noise only')
    plt.axvline(threshold, c='C2', ls='--', aa=True, label='Threshold')

    # print vertical line at means
    #     plt.axvline(np.mean(eng_both), c='C0', ls='--', aa=True)
    #     plt.axvline(np.mean(eng_noise), c='C1', ls='--', aa=True)

    # x values for pdfs
    x = np.arange(int(np.amin(np.concatenate((eng_both, eng_noise)))),
                  int(np.amax(np.concatenate((eng_both, eng_noise)))))

    # compute powers
    noise_pow = util.dB_to_factor_power(noise_power)
    both_pow = noise_pow + util.dB_to_factor_power(signal_power)

    # CLT pdfs
    noise_dist = stats.norm.pdf(x,
                                loc=n * noise_pow,
                                scale=np.sqrt(n * noise_pow**2))
    both_dist = stats.norm.pdf(x,
                               loc=n * both_pow,
                               scale=np.sqrt(n * both_pow**2))
    plt.plot(x, both_dist, c='C3', ls='-', aa=True, label='CLT sig present')
    plt.plot(x, noise_dist, c='C4', ls='-', aa=True, label='CLT noise only')

    # Chi2 pdfs
    #     noise_dist = stats.chi2.pdf(x, df=2. * n, loc=0., scale=noise_pow / 2.)
    #     both_dist = stats.chi2.pdf(x, df=2. * n, loc=0., scale=both_pow / 2.)
    #     plt.plot(x, both_dist, c='C3', ls='-', aa=True, label='Chi2 sig present')
    #     plt.plot(x, noise_dist, c='C4', ls='-', aa=True, label='Chi2 noise only')

    plt.legend(loc=0)
    plt.grid(linewidth=0.5)
    plt.xlabel('Energy')
    plt.ylabel('Percentages')
    plt.show()

    print('---- Distribution stats ----')
    print('Sig present mean %.2f' % (np.mean(eng_both)))
    print('Sig absent  mean %.2f' % (np.mean(eng_noise)))
    print('Sig present var  %.2f' % (np.var(eng_both)))
    print('Sig absent  var  %.2f' % (np.var(eng_noise)))
    print('Sig present std  %.2f' % (np.std(eng_both)))
    print('Sig absent  std  %.2f' % (np.std(eng_noise)))
