from ray import tune

dqn_search_space = {
    'noisy': False,
    'lr': tune.uniform(lower=5e-08, upper=1e-1),
    'sigma0': tune.uniform(lower=0.0, upper=2.0),
    'dueling': True,
    'hiddens': tune.choice(([128, 128], [256, 128], [256, 256], [512, 256],
                            [256, 512], [256], [512], [128], [64])),
    'replay_buffer_config': {
        'capacity': tune.randint(lower=1500, upper=70000),
        'prioritized_replay_alpha': tune.uniform(lower=0.3, upper=0.7),
        'prioritized_replay_beta': tune.uniform(lower=0.2, upper=0.5)
    },
    'double_q': True,
    'n_step': 1,
    'categorical_distribution_temperature': tune.uniform(lower=0.0, upper=2.0),
    'train_batch_size': tune.randint(lower=28, upper=400),
}
