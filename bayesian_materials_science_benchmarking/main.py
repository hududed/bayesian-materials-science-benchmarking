import copy
import random
import pandas as pd
import numpy as np
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


def main() -> None:
    unique_ds, feature_name, objective_name = load_data('Crossed barrel')
    X_feature = unique_ds[feature_name].values
    y = np.array(unique_ds[objective_name].values)
    n_dataset = len(unique_ds)
    print(f"Number of data in set: {n_dataset}")

    # number of top candidates, currently using top 5% of total dataset size
    n_top = int(math.ceil(len(y)*.05))
    top_indices = list(unique_ds.sort_values(objective_name).head(n_top).index)

    print(top_indices)

    # time_before = perf_counter()
    # # these will carry results along optimization sequence from all n_ensemble runs
    # index_collection = []
    # X_collection = []
    # y_collection = []
    # TopCount_collection = []

    # indices = list(np.arange(n_dataset))
    # index_learn = indices.copy()
    # index_ = random.sample(index_learn, n_init)
    # print(index_)


if __name__ == "__main__":
    main()
