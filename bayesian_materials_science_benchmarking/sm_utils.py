import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor



def RF_pred(X: np.ndarray, RF_model: RandomForestRegressor, n_est: int) -> tuple[float, float]:
    tree_predictions = []
    for j in np.arange(n_est):
        tree_predictions.append((RF_model.estimators_[j].predict(np.array([X]))).tolist())
    mean = np.mean(np.array(tree_predictions), axis=0)[0]
    
    std = np.std(np.array(tree_predictions), axis=0)[0]
    return mean, std


def EI(X: np.ndarray, RF_model: RandomForestRegressor, y_best: float | np.ndarray) -> float:
    mean, std = RF_pred(X, RF_model)
    z = (y_best - mean)/std
    return (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)

def LCB(X: np.ndarray, RF_model: RandomForestRegressor, ratio: float) -> float:
    mean, std = RF_pred(X, RF_model)
    return - mean + ratio * std

def PI(X: np.ndarray, RF_model: RandomForestRegressor, y_best: float | np.ndarray) -> float:
    mean, std = RF_pred(X, RF_model)
    z = (y_best - mean)/std
    return norm.cdf(z)