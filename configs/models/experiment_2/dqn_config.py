
dqn_config = {
    'hiddens': [256, 256],
    'noisy': False,
    'lr': 0.00023905045501405943,
    'sigma0': 1.764313877579869,
    'dueling': True,
    'double_q': True,
    'train_batch_size': 211,
    'exploration_config': {
        'epsilon_timesteps': 14998,
        'final_epsilon': 0.05,
        'initial_epsilon': 1.0
    },
    'replay_buffer_config': {'capacity': 49766, 'prioritized_replay_alpha': 0.6163607766186519,
                             'prioritized_replay_beta': 0.41847997435699963}
}
