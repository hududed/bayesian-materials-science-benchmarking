from time import perf_counter
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from model_utils import EI, PI, LCB


def train_ensemble(
        n_ensemble: int, n_init: int, n_dataset: int, n_est: int, X_feature: np.ndarray,
        y: np.ndarray, top_indices: list[int], seed_list: list[int], dataset_name: str,
        acq_func: str, ratio: float = 0.2
) -> None:
    time_before = perf_counter()

    if acq_func == "LCB":
        print(f"+++ {acq_func}{ratio}")
    else:
        print(f"+++ {acq_func}")

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

        # for each of the the rest of (N - n_initial) learning cycles
        # this for loop ends when all candidates in pool are observed
        for idx, i in enumerate(np.arange(len(index_learn))):

            y_best = np.min(y_observed)

            s_scaler = StandardScaler()

            X_train = s_scaler.fit_transform(X_observed)
            y_train = s_scaler.fit_transform([[i] for i in y_observed])

            print(
                f"FITTING: {idx} / {len(indices)} BEST Y: {y_best}")

            RF_model = RandomForestRegressor(n_estimators=n_est, n_jobs=-1)
            RF_model.fit(X_train, y_train.ravel())

        # by evaluating acquisition function values at candidates remaining in pool
        # we choose candidate with larger acquisition function value to be observed next
            next_index = None
            max_ac = -10**10
            for idx, j in enumerate(index_learn):
                X_j = X_feature[j]
                y_j = y[j]
                # TODO: select Acquisiton Function for BO
                if acq_func == 'EI':
                    ac_value, ac_name = EI(X_j, RF_model, n_est, y_best)
                if acq_func == 'PI':
                    ac_value, ac_name = PI(X_j, RF_model, n_est, y_best)
                if acq_func == 'LCB':
                    ac_value, ac_name = LCB(X_j, RF_model, n_est, ratio)

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
            f"### COLL IDX: {len(index_collection)}\n X: {len(X_collection)}\n y: {len(y_collection)}\n top: {len(TopCount_collection)} ")

        print('Finished seed')

    total_time = perf_counter() - time_before

    master = np.array([index_collection, X_collection,
                      y_collection, TopCount_collection, total_time], dtype=object)

    if acq_func == 'LCB':
        np.save(f"{ac_name}{ratio}_{dataset_name}", master)
    else:
        np.save(f"{ac_name}_{dataset_name}", master)
