import pandas as pd
from perf_utils import avg_, TopPercent
from config import DATA_PATH, acq_funcs
from copy import deepcopy
from typing import Dict, Any
from collections import defaultdict
import numpy as np


def load_data(name: str, invert_y: bool) -> tuple[pd.DataFrame, list[str], str]:
    raw_dataset = pd.read_csv(DATA_PATH / f"{name}_dataset.csv")
    feature_name = list(raw_dataset.columns)[:-1]
    objective_name = list(raw_dataset.columns)[-1]
    ds = deepcopy(raw_dataset)
    if invert_y:
        ds[objective_name] = -raw_dataset[objective_name].values
    unique_ds = ds.groupby(feature_name)[objective_name].agg(
        lambda x: x.unique().mean())
    unique_ds = (unique_ds.to_frame()).reset_index()
    # debug
    # unique_ds = unique_ds.iloc[:5]

    return unique_ds, feature_name, objective_name

# def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
#     s_scaler = StandardScaler()
#     df_normalized_values = s_scaler.fit_transform(ds_grouped[list(raw_dataset.columns)].values)
#     df_normalized = pd.DataFrame(df_normalized_values, columns = list(raw_dataset.columns))
#     return ds_normalized


def categorical_to_int(df: pd.DataFrame, col_name: str, categories: Dict[str, Any]) -> pd.DataFrame:
    df[col_name] = df[col_name].replace(categories)
    return df


def create_results_dict_for(dataset_name: str, n_top: int, n_dataset: int) -> Dict[str, Any]:
    d = defaultdict(list)
    for af_name in acq_funcs:
        results = np.load(f"{af_name}_{dataset_name}.npy", allow_pickle=True)
        d[af_name] = avg_(TopPercent(results[3], n_top, n_dataset))
    return dict(d)
