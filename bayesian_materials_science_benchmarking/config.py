import math
from pathlib import Path
import numpy as np

DATA_PATH = Path('datasets')
FIG_PATH = Path('figures')


# benchmarking framework params
n_ensemble = 50
n_init = 2
rng = np.random.default_rng(2021)
seed_list = list(rng.integers(1000, size=n_ensemble))

# random forest params
n_est = 100

# data preprocessing params
categories = {
    'Argon': 1,
    'Nitrogen': 2,
    'Air': 3,
}
