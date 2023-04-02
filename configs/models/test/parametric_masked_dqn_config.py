from configs.models.test.dqn_config import dqn_config_oilfield

from src.algorithms.exploration.exploration import DynamicIntervalEpsilonGreedy

pam_config_test = {
    'dueling': True,
    'double_q': True,
    'exploration_config': {
        'type': DynamicIntervalEpsilonGreedy,
        'epsilon_timesteps': 21744,
        'final_epsilon': 0.0817,
        'initial_epsilon': 0.215
    },
    'hiddens': [278, 584],
    'lr': 0.051,
    'n_step': 1,
    'noisy': False,
    'normalize_actions': False,
    'p_lr': 0.000355,
    'q_lr': 0.000204,
    'train_batch_size': 205
}

pam_config_oilfield = {
    **dqn_config_oilfield,
    'exploration_config': {
        'type': DynamicIntervalEpsilonGreedy,
        "initial_epsilon": 1.0,
        "final_epsilon": 0.02,
        "epsilon_timesteps": 10000
    },
    'q_lr': 1e-5,
    'p_lr': 1e-5,
    'normalize_actions': False
}
