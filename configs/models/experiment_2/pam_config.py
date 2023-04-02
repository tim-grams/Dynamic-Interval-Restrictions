from src.algorithms.exploration.exploration import DynamicIntervalEpsilonGreedy

pam_config = {
    'dueling': True,
    'double_q': True,
    'exploration_config': {'type': DynamicIntervalEpsilonGreedy,
                           'epsilon_timesteps': 7157,
                           'final_epsilon': 0.013985350229506357,
                           'initial_epsilon': 0.6542201567070297
                           },
    'replay_buffer_config': {'capacity': 19299,
                             'prioritized_replay_alpha': 0.5509030368975296,
                             'prioritized_replay_beta': 0.2969250149040724},
    'q_hiddens': [64],
    'p_hiddens': [512, 56],
    'categorical_distribution_temperature': 0.5721090461439949,
    'sigma0': 0.9063566517922343,
    'lr': 0.011,
    'n_step': 1,
    'noisy': False,
    'normalize_actions': False,
    'p_lr': 0.00013595132183729435,
    'q_lr': 3.453960761764736e-05,
    'train_batch_size': 148
}
