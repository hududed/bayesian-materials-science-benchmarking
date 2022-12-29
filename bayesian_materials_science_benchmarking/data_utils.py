import pandas as pd
from config import DATA_PATH
from copy import deepcopy
from typing import Dict, Any


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


def categorical_to_int(df: pd.DataFrame, col_name: str, categories: Dict[str, Any]) -> pd.DataFrame:
    df[col_name] = df[col_name].replace(categories)
    return df
