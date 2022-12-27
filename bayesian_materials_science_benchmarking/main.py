import pandas as pd
from config import DATA_PATH

def main() -> None:
    dataset_name = 'Crossed barrel'
    raw_dataset = pd.read_csv(DATA_PATH / f"{dataset_name}_dataset.csv")
    print(raw_dataset.head())
if __name__ == "__main__":
    main()