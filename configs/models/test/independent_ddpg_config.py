from src.algorithms.exploration.exploration import DynamicIntervalGaussianNoise

from configs.models.test.td3_config import td3_config_test, td3_config_oilfield

mps_td3_config_test = {
    **td3_config_test,
    'exploration_config': {
        'type': DynamicIntervalGaussianNoise,
        'random_timesteps': 1000,
        'initial_scale': 10.0,
        'final_scale': 0.02,
        'scale_timesteps': 20000
    }
}

mps_td3_config_oilfield = {
    **td3_config_oilfield,
    'exploration_config': {
        'type': DynamicIntervalGaussianNoise,
        'random_timesteps': 1000,
        'initial_scale': 10.0,
        'final_scale': 0.02,
        'scale_timesteps': 20000
    }
}
