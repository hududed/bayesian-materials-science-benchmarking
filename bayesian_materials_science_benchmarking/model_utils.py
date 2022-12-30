import inspect
from typing import Callable
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor


def RF_pred(X: np.ndarray, RF_model: RandomForestRegressor, n_est: int = 100) -> tuple[float, float]:
    tree_predictions = []
    for j in np.arange(n_est):
        tree_predictions.append(
            (RF_model.estimators_[j].predict(np.array([X]))).tolist())
    mean = np.mean(np.array(tree_predictions), axis=0)[0]

    std = np.std(np.array(tree_predictions), axis=0)[0]
    return mean, std


def EI(X: np.ndarray, RF_model: RandomForestRegressor, n_est: int, y_best: float) -> tuple[float, str]:
    mean, std = RF_pred(X, RF_model, n_est)
    with np.errstate(divide='raise'):
        try:
            z = (y_best - mean)/std
        except FloatingPointError:  # divide by zero numpy warning
            std = 1e-16
            z = (y_best - mean)/std
            print(f"\n !! DIVIDE BY ZERO {mean} {std} {z}")
    func_name = inspect.currentframe().f_code.co_name
    return (y_best - mean) * norm.cdf(z) + std * norm.pdf(z), func_name


def PI(X: np.ndarray, RF_model: RandomForestRegressor, n_est: int, y_best: float) -> tuple[float, str]:
    mean, std = RF_pred(X, RF_model, n_est)
    with np.errstate(divide='raise'):
        try:
            z = (y_best - mean)/std
        except FloatingPointError:
            std = 1e-16
            z = (y_best - mean)/std
            print(f"\n !! DIVIDE BY ZERO {mean} {std} {z}")
    func_name = inspect.currentframe().f_code.co_name
    return norm.cdf(z), func_name


def LCB(X: np.ndarray, RF_model: RandomForestRegressor,  n_est: int, ratio: float = 0.5) -> tuple[float, str]:
    mean, std = RF_pred(X, RF_model, n_est)
    func_name = inspect.currentframe().f_code.co_name
    return - mean + ratio * std, func_name


# RandomForestAcqFuncFunction = Callable[[
#     np.ndarray, RandomForestRegressor, int], tuple[float, str]]


# AF_PROCESSORS = [
#     EI, PI
# ]


# def handle_RF_acquisition_processors(
#         X: np.ndarray, RF_model: RandomForestRegressor, n_est: int,
#         af_processors: list[RandomForestAcqFuncFunction]
# ) -> None:
#     for af_processor in af_processors:
#         af_processor(X, RF_model, n_est)
