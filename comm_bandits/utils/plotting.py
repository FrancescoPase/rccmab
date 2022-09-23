import matplotlib.pyplot as plt
import numpy as np


def plot_distributions(xs, qs, ps=None, us=None):
    if us is None and ps is None:
        fig = plt.figure(figsize=(6, 4))
        plt.title('Distribution')
        plt.plot(xs, qs, marker='*', color='blue')
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.ylim([0, 1])
        plt.plot()
    else:
        maximum = np.max(np.hstack([qs, ps, us]))
        fig, axs = plt.subplots(1, 3, figsize=(24, 4))

        axs[0].set_title('Adopted Policy', fontsize=23)
        axs[0].plot(xs, qs, marker='*', color='green')
        # axs[0].set_xlabel('a',  fontsize=23)
        axs[0].set_ylabel('q (a|s)', fontsize=23)
        axs[0].tick_params(axis="x", labelsize=20)
        axs[0].tick_params(axis="y", labelsize=20)
        axs[0].set_ylim([0, 1])

        axs[1].set_title('Target Policy', fontsize=23)
        axs[1].plot(xs, ps, marker='*', color='red')
        # axs[1].set_xlabel('a',  fontsize=23)
        axs[1].tick_params(axis="x", labelsize=20)
        axs[1].tick_params(axis="y", labelsize=20)
        axs[1].set_ylabel('π (a|s)', fontsize=23)
        axs[1].set_ylim([0, 1])

        axs[2].set_title('Marginal', fontsize=23)
        axs[2].plot(xs, us, marker='*', color='blue')
        axs[2].tick_params(axis="x", labelsize=20)
        axs[2].tick_params(axis="y", labelsize=20)
        # axs[2].set_xlabel('a',  fontsize=23)
        axs[2].set_ylabel('q (a)', fontsize=23)
        axs[2].set_ylim([0, 1])

        plt.plot()
