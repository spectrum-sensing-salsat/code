import numpy as np
import matplotlib.pyplot as plt
import time


def runtime_stats(current_time, total_itr, current_itr):
    if current_time is None:  # first iteration cant predict time
        current_time = time.time()
        return float('inf'), 0.0, current_time
    delta_time = time.time() - current_time
    current_time = time.time()
    remaining_itr = total_itr - current_itr
    remaining_time = delta_time * remaining_itr
    percent_done = current_itr / total_itr * 100.0
    return remaining_time, percent_done, current_time


def print_convergence(gens, pfas, pds):
    plt.figure(figsize=(8, 6))
    plt.grid(linewidth=0.3)
    for i in range(1, gens):
        inter = np.mean(pfas[0:i])
        plt.plot(i, inter, 'ko', markersize=1, aa=True)
    plt.axhline(np.mean(pfas), xmin=0., xmax=1., color='k', ls='--', alpha=0.5)
    plt.xlabel(r'Generations')
    plt.ylabel(r'$P_{fa}$ (Probability of false alarm)')
    plt.show()
    plt.figure(figsize=(8, 6))
    plt.grid(linewidth=0.3)
    for i in range(1, gens):
        inter = np.mean(pds[0:i])
        plt.plot(i, inter, 'ko', markersize=1, aa=True)
    plt.axhline(np.mean(pds), xmin=0., xmax=1., color='k', ls='--', alpha=0.5)
    plt.xlabel(r'Generations')
    plt.ylabel(r'$P_d$ (Probability of detection)')
    plt.show()


def print_distibution(eng_sig, eng_noise, bins=100):
    plt.figure(figsize=(8, 6))
    plt.title(r'Distribution when signal present')
    plt.hist(eng_sig, bins=bins)
    plt.show()
    print('Sig Mean=%.3f, Standard Deviation=%.3f' %
          (np.mean(eng_sig), np.std(eng_sig)))

    plt.title(r'Distribution when signal absent')
    plt.hist(eng_noise, bins=bins)
    plt.show()
    print('Noise Mean=%.3f, Standard Deviation=%.3f' %
          (np.mean(eng_noise), np.std(eng_noise)))
