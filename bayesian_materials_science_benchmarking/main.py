import numpy as np
from data_utils import load_data, categorical_to_int
from config import n_ensemble, n_init, n_est, seed_list, ratios, categories, tops
import math
from train_utils import train_ensemble


def main() -> None:

    dataset_name = 'Perovskite'
    unique_ds, feature_name, objective_name = load_data(
        dataset_name, invert_y=False)

    print(f"##### START: {dataset_name} #####\n")
    # unique_ds = categorical_to_int(
    #     unique_ds, col_name='gas', categories=categories)
    X_feature = unique_ds[feature_name].values
    y = np.array(unique_ds[objective_name].values)
    n_dataset = len(unique_ds)
    print(f"Number of data in set: {n_dataset}")

    # number of top candidates, currently using top 20% of total dataset size
    top = tops[-1]
    n_top = int(math.ceil(len(y)*top))
    top_indices = list(unique_ds.sort_values(objective_name).head(n_top).index)

    train_ensemble(n_ensemble, n_init, n_dataset, n_est, X_feature,
                   y, top_indices, seed_list, dataset_name, acq_func='EI', top=top)

    train_ensemble(n_ensemble, n_init, n_dataset, n_est, X_feature,
                   y, top_indices, seed_list, dataset_name, acq_func='PI', top=top)

    for ratio in ratios:
        train_ensemble(
            n_ensemble, n_init, n_dataset, n_est, X_feature,
            y, top_indices, seed_list, dataset_name, acq_func='LCB', top=top, ratio=ratio)


if __name__ == "__main__":
    main()
