import copy
import pandas as pd
from config import *



def main() -> None:
    dataset_name = 'Crossed barrel'
    raw_dataset = pd.read_csv(DATA_PATH / f"{dataset_name}_dataset.csv")
    feature_name = list(raw_dataset.columns)[:-1]
    objective_name = list(raw_dataset.columns)[-1]
    ds = copy.deepcopy(raw_dataset)
    ds[objective_name] = -raw_dataset[objective_name].values

    unique_ds = ds.groupby(feature_name)[objective_name].agg(lambda x: x.unique().mean())
    unique_ds = (unique_ds.to_frame()).reset_index()

    X_feature = unique_ds[feature_name].values
    y = np.array(unique_ds[objective_name].values)

    n_top = int(math.ceil(len(y)*.05))
    top_indices = list(unique_ds.sort_values(objective_name).head(n_top).index)

    



if __name__ == "__main__":
    main()