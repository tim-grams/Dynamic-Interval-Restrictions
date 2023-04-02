from src.utils.utils import sample_from_gaussian_process
from sklearn.gaussian_process.kernels import RBF
import numpy as np

x_sample, y_sample = sample_from_gaussian_process(0.0, 40.0, 600, RBF(3), 18)

oil_field_config = {
    'LENGTH': 40.0,
    'LENGTH_PUMP': 2.0,
    'LENGTH_PUMP_STD': 1.0,
    'LENGTH_PUMP_MIN': 1.0,
    'LENGTH_PUMP_MAX': 3.0,
    'STEPS_PER_EPISODE': 100,
    'EFFECTIVENESS_MEAN': 1.0,
    'EFFECTIVENESS_STD': 0.3,
    'EFFECTIVENESS_MIN': 0.5,
    'EFFECTIVENESS_MAX': 2.0,
    'DURATION_MEAN': 3,
    'DURATION_MIN': 1,
    'DURATION_MAX': 5,
    'GAUSSIAN_NOISE': 0.2,
    'valid_action_reward': lambda x: np.interp(x, x_sample, y_sample.T[0]),
    'invalid_action_penalty': lambda x: -10 * x,
    'END_ON_COLLISION': False
}
