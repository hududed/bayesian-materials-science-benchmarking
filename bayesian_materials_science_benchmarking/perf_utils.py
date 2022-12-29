import math
import numpy as np
from config import seed_list


def perf_random(n_dataset: int, n_top: int):
    """ Random baseline"""
    x_random = np.arange(n_dataset)

    M = n_top
    N = n_dataset

    P = np.array([None for i in x_random])
    E = np.array([None for i in x_random])
    A = np.array([None for i in x_random])
    cA = np.array([None for i in x_random])

    P[0] = M / N
    E[0] = M / N
    A[0] = M / N
    cA[0] = A[0]

    for i in x_random[1:]:
        P[i] = (M - E[i-1]) / (N - i)
        E[i] = np.sum(P[:(i+1)])
        j = 0
        A_i = P[i]
        while j < i:
            A_i *= (1 - P[j])
            j += 1
        A[i] = A_i
        cA[i] = np.sum(A[:(i+1)])

    return E / M, cA


def aggregation_(seed: int, n_runs: int, n_fold: int) -> list[np.ndarray]:

    assert math.fmod(n_runs, n_fold) == 0
    fold_size = int(n_runs / n_fold)

    rng = np.random.default_rng(seed)

    index_runs = list(np.arange(n_runs))

    agg_list = []

    i = 0

    while i < n_fold:
        index_i = rng.choice(index_runs, fold_size, replace=False)
        # print(f"index i: {index_i}")
        for j in index_i:
            # print(j)
            index_runs.remove(j)
            # print(f"indexrun: {index_runs}")

        agg_list.append(index_i)

        i += 1
#     print(agg_list)
    return agg_list


def avg_(x: list[list[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     nsteps
    n_eval = len(x[0])

    #     fold
    n_fold = 5

    #     rows = # of ensembles = 50
    n_runs = len(x)

    assert math.fmod(n_runs, n_fold) == 0
    fold_size = int(n_runs / n_fold)

    #     # of seeds
    n_sets = len(seed_list)

    l_index_list = []

    for i in np.arange(n_sets):

        s = aggregation_(seed_list[i], n_runs, n_fold)
        l_index_list.extend(s)

    #     rows in l_index_list

    assert len(l_index_list) == n_sets * n_fold

    l_avg_runs = []

    for i in np.arange(len(l_index_list)):

        avg_run = np.zeros(n_eval)
        for j in l_index_list[i]:

            avg_run += np.array(x[j])

        avg_run = avg_run/fold_size
        l_avg_runs.append(avg_run)

    assert n_eval == len(l_avg_runs[0])
    assert n_sets * n_fold == len(l_avg_runs)

    mean_ = [None for i in np.arange(n_eval)]
    std_ = [None for i in np.arange(n_eval)]
    median_ = [None for i in np.arange(n_eval)]
    low_q = [None for i in np.arange(n_eval)]
    high_q = [None for i in np.arange(n_eval)]


#     5th, 95th percentile, mean, median are all accessible
    for i in np.arange(len(l_avg_runs[0])):
        i_column = []
        for j in np.arange(len(l_avg_runs)):
            i_column.append(l_avg_runs[j][i])

        i_column = np.array(i_column)
        mean_[i] = np.mean(i_column)
        median_[i] = np.median(i_column)
        std_[i] = np.std(i_column)
        low_q[i] = np.quantile(i_column, 0.05, out=None,
                               overwrite_input=False, interpolation='linear')
        high_q[i] = np.quantile(i_column, 0.95, out=None,
                                overwrite_input=False, interpolation='linear')

    return np.array(median_), np.array(low_q), np.array(high_q), np.array(mean_), np.array(std_)


def TopPercent(x_top_count: list[list[float]], n_top: int, n_dataset: int) -> list[list[float]]:

    x_ = [[] for i in np.arange(len(x_top_count))]

    for i in np.arange(len(x_top_count)):
        for j in np.arange(n_dataset):
            if j < len(x_top_count[i]):
                x_[i].append(x_top_count[i][j] / n_top)
            else:
                x_[i].append(1.0)

    return x_


def EF(x: np.ndarray, n_top: int):
    n_eval = len(x)
    TopPercent_RS = perf_random(n_eval, n_top)[0]

    l_EF = []
    for j in np.arange(n_eval):
        l_EF.append(x[j] / TopPercent_RS[j])

    return l_EF
