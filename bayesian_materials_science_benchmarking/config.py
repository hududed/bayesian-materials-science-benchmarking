from enum import Enum, auto
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


class Categories(Enum):
    ARGON = auto()
    NITROGEN = auto()
    AIR = auto()


categories = {
    'Argon': Categories.ARGON.value,
    'Nitrogen': Categories.NITROGEN.value,
    'Air': Categories.AIR.value
}


# acquisition func LCB params
ratios = [0.1, 0.2, 0.5, 1., 2., 5., 10.]

# list of acquisition functions
top = .05
raw_acq_funcs = [
    'EI', 'PI',
    'LCB0.1', 'LCB0.2', 'LCB0.5', 'LCB1.0',
    'LCB2.0',
    # 'LCB5.0',
    # 'LCB10.0'
]
acq_funcs = [f"{top}_{raw_acq_func}" for raw_acq_func in raw_acq_funcs]
