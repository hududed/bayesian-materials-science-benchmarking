from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np
from perf_utils import EF, AF_interp1d, perf_random
from config import FIG_PATH
from typing import Dict, Any


def plot_top_cycle(n_dataset: int, n_top: int, results_dict: Dict[str, tuple[np.ndarray,  np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dataset_name: str) -> None:

    fig = plt.figure(figsize=(12, 12))
    ax0 = fig.add_subplot(111)

    ax0.plot(np.arange(n_dataset)+1, perf_random(n_dataset, n_top)
             [0], '--', color='black', label='random baseline', linewidth=3.5)

    for af_name, aggr_results in results_dict.items():
        ax0.plot(np.arange(n_dataset) + 1, np.round(aggr_results[0].astype(
            np.double) / 0.005, 0) * 0.005, label=f'RF : {af_name}', linewidth=3)
        ax0.fill_between(np.arange(n_dataset) + 1, np.round(aggr_results[1].astype(np.double) / 0.005, 0) * 0.005, np.round(
            aggr_results[2].astype(np.double) / 0.005, 0) * 0.005, alpha=0.2)

    # the rest are for visualization purposes, please adjust for different needs
    n_plots = len(results_dict)
    cmap = mpl.colormaps["tab10"].resampled(n_plots)
    plt.gca().set_prop_cycle(cycler('color', cmap(np.linspace(0, 1, n_plots))))
    font = font_manager.FontProperties(
        family='Arial', size=26, style='normal')
    leg = ax0.legend(prop=font, borderaxespad=0,  labelspacing=0.3,
                     handlelength=1.2, handletextpad=0.3, frameon=False, loc=(0, 0.81))
    for line in leg.get_lines():
        line.set_linewidth(4)
    ax0.set_ylabel("Top%", fontname="Arial",
                   fontsize=30, rotation='vertical')
    plt.hlines(0.8, 0, 480, colors='k', linestyles='--', alpha=0.2)
    # ax0.set_xlim([0, 300])
    ax0.set_ylim([0, 1.05])
    ax0.set_xscale('log')
    ax0.set_xlabel('learning cycle $i$', fontsize=30, fontname='Arial')
    ax0.xaxis.set_tick_params(labelsize=30)
    ax0.yaxis.set_tick_params(labelsize=30)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    # plt.xticks([1, 2, 10, 100, 600], ['1', '2', '10',
    #            '10$^{\mathrm{2}}$', '6$×$10$^{\mathrm{2}}$'], fontname='Arial')
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2',
    #            '0.4', '0.6', '0.8', '1.0'], fontname='Arial')

    plt.savefig(
        FIG_PATH / f'top_{dataset_name}.png', dpi=300, format="png")


def plot_EF(n_dataset: int, n_top: int, results_dict: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dataset_name: str) -> None:

    fig = plt.figure(figsize=(12, 12))
    ax0 = fig.add_subplot(111)

    ax0.plot(np.linspace(1, n_dataset, n_dataset), np.ones(n_dataset),
             '--', color='black', label='random baseline', linewidth=3)

    for af_name, aggr_results in results_dict.items():
        ax0.plot(np.arange(n_dataset) + 1, EF(np.round(aggr_results[0].astype(
            np.double) / 0.005, 0) * 0.005, n_top), label=f'RF : {af_name}', linewidth=3)
        ax0.fill_between(np.arange(n_dataset) + 1, EF(np.round(aggr_results[1].astype(np.double) / 0.005, 0) * 0.005, n_top), EF(
            np.round(aggr_results[2].astype(np.double) / 0.005, 0) * 0.005, n_top),  alpha=0.2)

    # the rest are for visualization purposes, please adjust for different needs
    n_plots = len(results_dict)
    cmap = mpl.colormaps["tab10"].resampled(n_plots)
    plt.gca().set_prop_cycle(cycler('color', cmap(np.linspace(0, 1, n_plots))))
    ax0.set_ylabel('EF', fontsize=30, rotation='horizontal',
                   fontname='Arial', labelpad=10)
    ax0.set_xlabel('learning cycle $i$', fontsize=30, fontname='Arial')
    # ax0.set_xlim([0, 300])
    ax0.set_ylim([0, 2])
    ax0.set_xscale('log')
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.xaxis.set_tick_params(labelsize=30)
    ax0.yaxis.set_tick_params(labelsize=30)
    # plt.xticks([10, 100, 600], [
    #            '10', '10$^{\mathsf{2}}$', '6$×$10$^{\mathsf{2}}$'], fontname='Arial')
    # plt.yticks([0, 1, 2, 4, 6, 8, 10], ['0', '1', '2',
    #            '4', '6', '8', '10'], fontname='Arial')

    plt.savefig(
        FIG_PATH / f'EF_{dataset_name}.png', dpi=300, format="png")


def plot_AF(n_top: int, results_dict: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dataset_name: str) -> None:
    fig = plt.figure(figsize=(12, 12))
    ax0 = fig.add_subplot(111)

    ax0.plot(np.linspace(0, 1, 200), np.ones(200), '--',
             color='black', label=None, linewidth=3)

    for af_name, aggr_results in results_dict.items():
        xx_, f_med_, f_low_, f_high_ = AF_interp1d(n_top, aggr_results)
        ax0.plot(xx_, f_med_(
            xx_), label=f'RF : {af_name}',  linewidth=3)
        ax0.fill_between(xx_, f_low_(xx_), f_high_(xx_), alpha=0.2)

    # the rest are for visualization purposes, please adjust for different needs
    n_plots = len(results_dict)
    cmap = mpl.colormaps["tab10"].resampled(n_plots)
    plt.gca().set_prop_cycle(cycler('color', cmap(np.linspace(0, 1, n_plots))))
    ax0.set_ylabel('AF', fontsize=30, rotation='horizontal',
                   fontname='Arial', labelpad=10)
    ax0.set_xlabel('Top%', fontsize=30, fontname='Arial')
    # ax0.set_xlim([100, 300])
    ax0.set_ylim([0, 2])
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.xaxis.set_tick_params(labelsize=30)
    ax0.yaxis.set_tick_params(labelsize=30)
    # plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0', '0.2',
    #            '0.4', '0.6', '0.8', '1.0'], fontname='Arial')
    # plt.yticks([0, 1, 2, 4, 6, 8, 10], ['0', '1', '2',
    #            '4', '6', '8', '10'], fontname='Arial')
    plt.savefig(
        FIG_PATH/f'AF_{dataset_name}.png', dpi=300, format="png")
