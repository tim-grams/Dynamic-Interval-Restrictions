from ray import tune

from configs.search_spaces.td3_space import td3_search_space
from src.algorithms.exploration.exploration import DynamicIntervalGaussianNoise

mps_td3_search_space = {
    **td3_search_space,
    'exploration_config': {
        'type': DynamicIntervalGaussianNoise,
        'random_timesteps': tune.randint(lower=100, upper=400),
        'initial_scale': tune.uniform(lower=0.2, upper=30.0),
        'final_scale': tune.uniform(lower=0.01, upper=3.0),
        'scale_timesteps': tune.randint(lower=1000, upper=50000),
        "initial_epsilon": tune.uniform(lower=0.05, upper=0.4),
        "final_epsilon": tune.uniform(lower=0.005, upper=0.1),
        "epsilon_timesteps": tune.randint(lower=1000, upper=30000)
    }
}
