from src.algorithms.exploration.exploration import DynamicIntervalEpsilonGreedy

pam_config = {
    'dueling': True,
    'double_q': True,
    'exploration_config': {'type': DynamicIntervalEpsilonGreedy,
                           'epsilon_timesteps': 27070,
                           'final_epsilon': 0.0060403738274216814,
                           'initial_epsilon': 0.43469690691641283
                           },
    'replay_buffer_config': {'capacity': 2398,
                             'prioritized_replay_alpha': 0.5264928266492329,
                             'prioritized_replay_beta': 0.3638330255688689},
    'q_hiddens': [256],
    'p_hiddens': [256],
    'categorical_distribution_temperature': 1.5109290968657936,
    'sigma0': 1.8118710250394103,
    'lr': 0.011,
    'n_step': 1,
    'noisy': False,
    'normalize_actions': False,
    'p_lr': 0.0008072393989640188,
    'q_lr': 0.0009460124152963434,
    'train_batch_size': 237
}
