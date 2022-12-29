from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from perf_utils import EF, perf_random


def plot_top_cycle(n_dataset: int, n_top: int, aggr_results: tuple[np.ndarray,  np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:

    fig = plt.figure(figsize=(12, 12))
    ax0 = fig.add_subplot(111)

    ax0.plot(np.arange(n_dataset)+1, perf_random(n_dataset, n_top)
             [0], '--', color='black', label='random baseline', linewidth=3.5)

    ax0.plot(np.arange(n_dataset) + 1, np.round(aggr_results[0].astype(
        np.double) / 0.005, 0) * 0.005, label='GP M52 : LCB$_{\overline{2}}$', color='#006d2c', linewidth=3)
    ax0.fill_between(np.arange(n_dataset) + 1, np.round(aggr_results[1].astype(np.double) / 0.005, 0) * 0.005, np.round(
        aggr_results[2].astype(np.double) / 0.005, 0) * 0.005, color='#006d2c', alpha=0.2)

    # the rest are for visualization purposes, please adjust for different needs
    font = font_manager.FontProperties(family='Arial', size=26, style='normal')
    leg = ax0.legend(prop=font, borderaxespad=0,  labelspacing=0.3,
                     handlelength=1.2, handletextpad=0.3, frameon=False, loc=(0, 0.81))
    for line in leg.get_lines():
        line.set_linewidth(4)
    ax0.set_ylabel("Top%", fontname="Arial", fontsize=30, rotation='vertical')
    plt.hlines(0.8, 0, 480, colors='k', linestyles='--', alpha=0.2)
    ax0.set_xlim([100, 300])
    ax0.set_ylim([0, 1.05])
    # ax0.set_xscale('log')
    ax0.set_xlabel('learning cycle $i$', fontsize=30, fontname='Arial')
    ax0.xaxis.set_tick_params(labelsize=30)
    ax0.yaxis.set_tick_params(labelsize=30)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    # plt.xticks([1, 2, 10, 100, 600], ['1', '2', '10',
    #            '10$^{\mathrm{2}}$', '6$×$10$^{\mathrm{2}}$'], fontname='Arial')
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2',
    #            '0.4', '0.6', '0.8', '1.0'], fontname='Arial')

    plt.savefig('top_cycle.png', dpi=300, format="png")


def plot_EF(n_dataset: int, n_top: int, aggr_results: tuple[np.ndarray,  np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:

    fig = plt.figure(figsize=(12, 12))
    ax0 = fig.add_subplot(111)

    ax0.plot(np.linspace(1, n_dataset, n_dataset), np.ones(n_dataset),
             '--', color='black', label='random baseline', linewidth=3)

    ax0.plot(np.arange(n_dataset) + 1, EF(np.round(aggr_results[0].astype(
        np.double) / 0.005, 0) * 0.005, n_top), label='GP M52 : LCB$_{\overline{2}}$', color='#006d2c', linewidth=3)
    ax0.fill_between(np.arange(n_dataset) + 1, EF(np.round(aggr_results[1].astype(np.double) / 0.005, 0) * 0.005, n_top), EF(
        np.round(aggr_results[2].astype(np.double) / 0.005, 0) * 0.005, n_top), color='#006d2c', alpha=0.2)

    # the rest are for visualization purposes, please adjust for different needs

    ax0.set_ylabel('EF', fontsize=30, rotation='horizontal',
                   fontname='Arial', labelpad=10)
    ax0.set_xlabel('learning cycle $i$', fontsize=30, fontname='Arial')
    ax0.set_xlim([100, 300])
    ax0.set_ylim([0, 2])
    # ax0.set_xscale('log')
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.xaxis.set_tick_params(labelsize=30)
    ax0.yaxis.set_tick_params(labelsize=30)
    # plt.xticks([10, 100, 600], [
    #            '10', '10$^{\mathsf{2}}$', '6$×$10$^{\mathsf{2}}$'], fontname='Arial')
    # plt.yticks([0, 1, 2, 4, 6, 8, 10], ['0', '1', '2',
    #            '4', '6', '8', '10'], fontname='Arial')

    plt.savefig('ef.png', dpi=300, format="png")
