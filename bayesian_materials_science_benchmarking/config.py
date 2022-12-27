import math
from pathlib import Path
import numpy as np

DATA_PATH = Path('datasets')


# benchmarking framework params
n_ensemble = 50
n_init = 2
rng = np.random.default_rng(2021)
seed_list = rng.random(n_ensemble)

# random forest params
n_est = 100