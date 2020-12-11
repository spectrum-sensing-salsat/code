import numpy as np
import matplotlib.pyplot as plt
import time


def runtime_stats(current_time, total_itr, current_itr):
    if current_time is None:  # First iteration cant predict time
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
    plt.title('Probability of false alarm')
    plt.grid(linewidth=0.3)
    for i in range(1, gens):
        inter = np.sum(pfas[0:i]) / i
        plt.plot(i, inter, 'kx')
    plt.xlabel('Generations')
    plt.ylabel('Probability')
    plt.show()
    plt.figure(figsize=(8, 6))
    plt.title('Probability of detection')
    plt.grid(linewidth=0.3)
    for i in range(1, gens):
        inter = np.sum(pds[0:i]) / i
        plt.plot(i, inter, 'kx')
    plt.xlabel('Generations')
    plt.ylabel('Probability')
    plt.show()


def print_distibution(en_sig, en_noise):
    plt.figure(figsize=(8, 6))
    plt.title('Distribution if signal present')
    plt.hist(en_sig, bins=100)
    plt.show()
    print('Sig Mean=%.3f, Standard Deviation=%.3f' %
          (np.mean(en_sig), np.std(en_sig)))

    plt.title('Distribution if signal absent')
    plt.hist(en_noise, bins=100)
    plt.show()
    print('Noise Mean=%.3f, Standard Deviation=%.3f' %
          (np.mean(en_noise), np.std(en_noise)))
