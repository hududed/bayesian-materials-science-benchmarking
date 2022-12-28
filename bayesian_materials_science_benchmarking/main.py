import copy
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sm_utils import EI
from config import DATA_PATH, n_ensemble, n_init, n_est, seed_list
import math
from time import perf_counter

# def train_ensemble()


def load_data(name: str, invert_y: bool = False) -> tuple[pd.DataFrame, list[str], str]:
    raw_dataset = pd.read_csv(DATA_PATH / f"{name}_dataset.csv")
    feature_name = list(raw_dataset.columns)[:-1]
    objective_name = list(raw_dataset.columns)[-1]
    ds = copy.deepcopy(raw_dataset)
    if invert_y:
        ds[objective_name] = -raw_dataset[objective_name].values
    unique_ds = ds.groupby(feature_name)[objective_name].agg(
        lambda x: x.unique().mean())
    unique_ds = (unique_ds.to_frame()).reset_index()

    return unique_ds, feature_name, objective_name


def train_ensemble(n_ensemble: int, n_init: int, n_dataset: int, X_feature: np.ndarray,
                   y: np.ndarray, top_indices: list[int], seed_list: list[int], dataset_name: str) -> None:
    time_before = perf_counter()
    # these will carry results along optimization sequence from all n_ensemble runs
    index_collection = []
    X_collection = []
    y_collection = []
    TopCount_collection = []

    for seed in seed_list:
        print(f"seed: {seed}")

        if len(index_collection) == n_ensemble:
            break

        print(
            f'initializing seed = {str(seed_list.index(seed))} / {n_ensemble}')
        rng = np.random.default_rng(seed)

        indices = list(np.arange(n_dataset))
        # index_learn is the pool of candidates to be examined
        index_learn = indices.copy()
        # index_ is the list of candidates we have already observed
        # adding in the initial experiments
        index_observed = list(rng.choice(index_learn, n_init))
        # index_ = random.sample(index_learn, n_init)
        print(index_observed)

        # list to store all observed good candidates' input feature X
        X_observed = []
        # list to store all observed good candidates' objective value y
        y_observed = []
        # number of top candidates found so far
        top_sofar = 0
        # list of cumulative number of top candidates found at each learning cycle
        top_count = []
        # add the first n_initial experiments to collection
        for i in index_observed:
            print(
                f"INITIAL length {len(index_learn)}, X_: {X_observed}, y_:  {y_observed}, \
                    nr_top_candidates: {top_sofar}, top count: {top_count}")
            X_observed.append(X_feature[i])
            y_observed.append(y[i])
            if i in top_indices:
                top_sofar += 1
            top_count.append(top_sofar)
            index_learn.remove(i)

    #     for each of the the rest of (N - n_initial) learning cycles
    #     this for loop ends when all candidates in pool are observed
        for idx, i in enumerate(np.arange(len(index_learn))):

            y_best = np.min(y_observed)

            s_scaler = preprocessing.StandardScaler()
            X_train = s_scaler.fit_transform(X_observed)
            y_train = s_scaler.fit_transform([[i] for i in y_observed])

            print(
                f"FITTING: {idx} / {len(indices)} BEST Y: {y_best}")

            RF_model = RandomForestRegressor(n_estimators=n_est, n_jobs=-1)
            RF_model.fit(X_train, y_train.ravel())

    #         by evaluating acquisition function values at candidates remaining in pool
    #         we choose candidate with larger acquisition function value to be observed next
            next_index = None
            max_ac = -10**10
            for idx, j in enumerate(index_learn):
                X_j = X_feature[j]
                y_j = y[j]
    #             #TODO: select Acquisiton Function for BO

                ac_value = EI(X_j, RF_model, y_best, n_est)
                # ac_value = LCB(X_j, RF_model, 10)

                if max_ac <= ac_value:
                    # print(f"old max_ac: {max_ac}, new ac: {ac_value}, next index: {j} X_j: {X_j} ")
                    max_ac = ac_value
                    next_index = j

            X_observed.append(X_feature[next_index])
            y_observed.append(y[next_index])

            if next_index in top_indices:
                top_sofar += 1
                print(
                    f">>> {next_index} in {top_indices} -> TOP COUNT: {top_sofar}")

            top_count.append(top_sofar)

            index_learn.remove(next_index)
            index_observed.append(next_index)

        assert len(index_observed) == n_dataset

        index_collection.append(index_observed)
        X_collection.append(X_observed)
        y_collection.append(y_observed)
        TopCount_collection.append(top_count)

        print(
            f"### COLL IDX: {index_collection}\n X: {X_collection}\n y: {y_collection}\n top: {TopCount_collection} ")

        print('Finished seed')

    total_time = perf_counter() - time_before

    master = np.array([index_collection, X_collection,
                      y_collection, TopCount_collection, total_time])
    #  #TODO: name output file
    np.save(f"EI_{dataset_name}", master)


def main() -> None:
    dataset_name = 'Crossed barrel'
    unique_ds, feature_name, objective_name = load_data(dataset_name)
    X_feature = unique_ds[feature_name].values
    y = np.array(unique_ds[objective_name].values)
    n_dataset = len(unique_ds)
    print(f"Number of data in set: {n_dataset}")

    # number of top candidates, currently using top 5% of total dataset size
    n_top = int(math.ceil(len(y)*.05))
    top_indices = list(unique_ds.sort_values(objective_name).head(n_top).index)

    train_ensemble(n_ensemble, n_init, n_dataset, X_feature,
                   y, top_indices, seed_list, dataset_name)


if __name__ == "__main__":
    main()
